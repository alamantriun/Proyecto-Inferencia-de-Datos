import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Rendimiento de Cacao (Colombia)",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Carga de Datos ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    mart_path = Path("data/processed/model_mart_cacao.csv")
    if not mart_path.exists():
        return None
    df = pd.read_csv(mart_path)
    
    # Asegurar tipos
    if "anio" in df.columns:
        df["anio"] = df["anio"].astype(int)
    
    return df

@st.cache_data
def load_ml_results():
    res_path = Path("reports/tables/resultados_ml_ablation.csv")
    if not res_path.exists():
        return None
    return pd.read_csv(res_path)

df = load_data()
ml_results = load_ml_results()

# ── Interfaz Principal ───────────────────────────────────────────────────────

st.title("🍫 Inferencia de Rendimiento Agrícola: Cacao en Colombia")
st.markdown("""
Esta aplicación permite explorar los resultados del modelo de inferencia predictiva para el cultivo de cacao. 
El sistema cruza bases de datos de la **UPRA, IDEAM, AGROSAVIA y FINAGRO** para entender la dinámica biológica y económica del cacao.
""")

if df is None:
    st.error("No se encontró el archivo `model_mart_cacao.csv`. Por favor, ejecuta el pipeline primero.")
    st.stop()

# ── Barra Lateral (Filtros) ──────────────────────────────────────────────────
st.sidebar.header("Filtros de Análisis")
departamentos = sorted(df["departamento"].dropna().unique())
depto_sel = st.sidebar.selectbox("Seleccione Departamento", departamentos)

municipios = sorted(df[df["departamento"] == depto_sel]["municipio"].dropna().unique())
muni_sel = st.sidebar.selectbox("Seleccione Municipio", municipios)

# ── Procesamiento de Datos del Municipio Seleccionado ────────────────────────
df_muni = df[(df["departamento"] == depto_sel) & (df["municipio"] == muni_sel)].sort_values("anio")

if df_muni.empty:
    st.warning("No hay datos históricos para este municipio.")
    st.stop()

# ── Pestañas de Navegación ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Inercia Histórica (EVA)", 
    "🧪 Perfil de Suelos (AGROSAVIA)", 
    "📊 Resultados Machine Learning",
    "📝 Conclusiones Finales"
])

# ── PESTAÑA 1: Histórico ──
with tab1:
    st.subheader(f"Historial de Rendimiento en {muni_sel.title()}")
    
    # Gráfica de línea
    fig_hist = px.line(
        df_muni, 
        x="anio", 
        y="rendimiento_t_ha", 
        markers=True,
        title="Toneladas por Hectárea (2007 - 2024)",
        labels={"anio": "Año", "rendimiento_t_ha": "Rendimiento (t/ha)"},
        color_discrete_sequence=["#1f77b4"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Métricas clave (último año disponible)
    ultimo_anio = df_muni["anio"].max()
    datos_recientes = df_muni[df_muni["anio"] == ultimo_anio].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Año Más Reciente", str(ultimo_anio))
    col2.metric("Rendimiento", f"{datos_recientes['rendimiento_t_ha']:.2f} t/ha")
    col3.metric("Área Cosechada", f"{datos_recientes['area_cosechada_ha']:,.0f} ha")
    col4.metric("Producción Total", f"{datos_recientes['produccion_t']:,.0f} t")

# ── PESTAÑA 2: Suelos ──
with tab2:
    st.subheader("Propiedades Químicas del Suelo")
    st.markdown("Promedios del municipio extraídos de las muestras de **AGROSAVIA**.")
    
    if "ph_media" in df_muni.columns and pd.notna(datos_recientes["ph_media"]):
        col1, col2, col3 = st.columns(3)
        col1.metric("pH", f"{datos_recientes['ph_media']:.2f}")
        col2.metric("Fósforo (P)", f"{datos_recientes.get('fosforo_ppm_media', 0):.2f} ppm")
        col3.metric("Potasio (K)", f"{datos_recientes.get('potasio_meq_media', 0):.2f} meq")
        
        st.info("💡 **Inferencia del Modelo:** El modelo de Machine Learning determinó que la disponibilidad de Calcio, Fósforo y Potasio son los factores biofísicos más determinantes para el rendimiento del cacao en Colombia.")
    else:
        st.warning("AGROSAVIA no tiene muestras de suelo reportadas para este municipio.")

# ── PESTAÑA 3: Machine Learning ──
with tab3:
    st.subheader("Machine Learning (CatBoost) vs Inercia")
    st.markdown("¿Puede la IA predecir mejor que la matemática simple?")
    
    if ml_results is not None:
        # Promediar el MAE por Fase
        mae_por_fase = ml_results.groupby("Fase")["MAE"].mean().reset_index()
        
        nombres_fases = {
            "A": "Baseline / Inercia (Target t-1)",
            "B": "+ Clima (IDEAM)",
            "C": "+ Suelos (AGROSAVIA)",
            "D": "+ Aptitud (UPRA)",
            "E": "+ Crédito (FINAGRO)",
            "F": "+ Mercado Internacional (FRED)"
        }
        mae_por_fase["Descripción"] = mae_por_fase["Fase"].map(nombres_fases)
        
        fig_mae = px.bar(
            mae_por_fase, 
            x="MAE", 
            y="Descripción", 
            orientation="h",
            title="Error Absoluto Medio (Menor es Mejor)",
            color="MAE",
            color_continuous_scale="reds"
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    else:
        st.warning("No se encontraron los resultados del modelo. Ejecuta 02_train_ml.py")

# ── PESTAÑA 4: Conclusiones ──
with tab4:
    st.subheader("Conclusiones Finales del Proyecto")
    st.markdown("""
    1. **El límite de la agregación:** Promediar datos a nivel municipal destruye la varianza necesaria para que los algoritmos de ML encuentren patrones agronómicos reales en el cacao.
    2. **La Inercia Gana:** Dado que el cacao es un cultivo permanente de largo ciclo (25 años), la inercia (rendimiento del año pasado) siempre vence al ML cuando no se tienen micro-datos de finca.
    3. **Política Pública:** Las variables climáticas municipales (lluvia) aportan casi cero información predictiva nueva; pero la **calidad del suelo (K, P, Ca)** es estructural. Invertir en fertilización mueve más la aguja que la infraestructura climática.
    4. **Cero Data Leakage:** El pipeline desarrollado respeta una validación estricta *Rolling-Origin*, asegurando que es imposible predecir el futuro usando información del futuro.
    """)
