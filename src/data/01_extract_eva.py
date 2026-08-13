"""
Paso 01-02: Descarga y limpieza de EVA (2007-2018 + 2019-2024) — filtrado para CACAO
=====================================================================================
Ejecutar:
    python src/data/01_extract_eva.py

Genera:
    data/raw/eva_historica_raw.csv
    data/raw/eva_reciente_raw.csv
    data/processed/eva_cacao.csv   (unificado, limpio, solo cacao)
"""

import sys
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.soda_client import SodaClient

# ── Columnas reales confirmadas con la API ─────────────────────────────────────
# EVA 2007-2018  (id: 2pnw-mmge)
COLS_HIST = (
    "departamento,municipio,cultivo,a_o,periodo,"
    "rea_sembrada_ha,rea_cosechada_ha,producci_n_t,rendimiento_t_ha,"
    "ciclo_de_cultivo"
)
RENAME_HIST = {
    "a_o": "anio",
    "rea_sembrada_ha": "area_sembrada_ha",
    "rea_cosechada_ha": "area_cosechada_ha",
    "producci_n_t": "produccion_t",
    "ciclo_de_cultivo": "ciclo_cultivo",
}

# EVA 2019-2024  (id: uejq-wxrr)
COLS_REC = (
    "departamento,municipio,cultivo,a_o,periodo,"
    "rea_sembrada,rea_cosechada,producci_n,rendimiento,"
    "ciclo_del_cultivo"
)
RENAME_REC = {
    "a_o": "anio",
    "rea_sembrada": "area_sembrada_ha",
    "rea_cosechada": "area_cosechada_ha",
    "producci_n": "produccion_t",
    "rendimiento": "rendimiento_t_ha",
    "ciclo_del_cultivo": "ciclo_cultivo",
}

NUMERIC_COLS = [
    "anio", "area_sembrada_ha", "area_cosechada_ha",
    "produccion_t", "rendimiento_t_ha",
]


# ── Descarga ───────────────────────────────────────────────────────────────────

def get_where_clause(cultivo: str) -> str:
    c_up = cultivo.upper()
    if c_up == "CAFE":
        return "UPPER(cultivo) LIKE '%CAFE%' OR UPPER(cultivo) LIKE '%CAFÉ%'"
    elif c_up == "PLATANO":
        return "UPPER(cultivo) LIKE '%PLATANO%' OR UPPER(cultivo) LIKE '%PLÁTANO%'"
    else:
        return f"UPPER(cultivo) LIKE '%{c_up}%'"

def download_eva_historica(cultivo: str, max_records=None) -> pd.DataFrame:
    print(f"\n=== EVA 2007-2018 ({cultivo.upper()}) ===")
    client = SodaClient("2pnw-mmge")
    df = client.extract_all(
        select=COLS_HIST,
        where=get_where_clause(cultivo),
        order="a_o,departamento,municipio",
        max_records=max_records,
    )
    if df.empty:
        print("⚠ Sin datos")
        return df
    df = df.rename(columns=RENAME_HIST)
    df["fuente_eva"] = "historica"
    return df


def download_eva_reciente(cultivo: str, max_records=None) -> pd.DataFrame:
    print(f"\n=== EVA 2019-2024 ({cultivo.upper()}) ===")
    client = SodaClient("uejq-wxrr")
    df = client.extract_all(
        select=COLS_REC,
        where=get_where_clause(cultivo),
        order="a_o,departamento,municipio",
        max_records=max_records,
    )
    if df.empty:
        print("⚠ Sin datos")
        return df
    df = df.rename(columns=RENAME_REC)
    df["fuente_eva"] = "reciente"
    return df


# ── Limpieza ───────────────────────────────────────────────────────────────────

def clean_eva(df: pd.DataFrame) -> pd.DataFrame:
    # Tipos numéricos
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Estandarizar strings (sin acentos, mayúsculas)
    for col in ["departamento", "municipio", "cultivo"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().str.upper()
                .str.normalize("NFKD")
                .str.encode("ascii", errors="ignore")
                .str.decode("ascii")
            )

    # Filtro calidad
    n0 = len(df)
    df = df[
        df["rendimiento_t_ha"].gt(0) &
        df["area_cosechada_ha"].gt(0) &
        df["anio"].notna()
    ].copy()
    print(f"  Calidad: {n0:,} → {len(df):,} ({n0 - len(df):,} eliminados)")

    # Duplicados
    key = ["departamento", "municipio", "cultivo", "anio", "periodo"]
    key = [c for c in key if c in df.columns]
    n_dup = df.duplicated(subset=key).sum()
    if n_dup:
        print(f"  Duplicados: {n_dup:,} → eliminando...")
        df = df.drop_duplicates(subset=key, keep="first")

    return df


# ── Auditoría ──────────────────────────────────────────────────────────────────

def audit(df: pd.DataFrame, label: str):
    print(f"\n── {label} ──")
    print(f"  Obs          : {len(df):,}")
    print(f"  Municipios   : {df['municipio'].nunique():,}")
    print(f"  Departamentos: {df['departamento'].nunique():,}")
    anios = sorted(df["anio"].dropna().astype(int).unique())
    print(f"  Años         : {anios}")
    rend = df["rendimiento_t_ha"]
    print(f"  Rendimiento  : media={rend.mean():.3f}  std={rend.std():.3f}"
          f"  min={rend.min():.3f}  max={rend.max():.3f}  t/ha")
    n_sin_lag = df.groupby("municipio")["anio"].nunique()
    print(f"  Munic. con ≥3 años: {(n_sin_lag >= 3).sum():,}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Leer el cultivo objetivo desde el archivo de configuración
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    print(f"Iniciando extracción de EVA para: {cultivo.upper()}")

    # Descargar
    df_hist = download_eva_historica(cultivo)
    df_rec = download_eva_reciente(cultivo)

    # Limpiar y recolectar
    dfs = []
    if not df_hist.empty:
        dfs.append(clean_eva(df_hist))
    if not df_rec.empty:
        dfs.append(clean_eva(df_rec))

    if not dfs:
        print("❌ Sin datos procesados.")
        return

    eva = pd.concat(dfs, ignore_index=True)
    eva["split"] = eva["anio"].apply(lambda y: "historial" if y <= 2018 else "target")

    # Guardar a disco
    proc = Path("data/processed")
    proc.mkdir(parents=True, exist_ok=True)
    
    file_name = f"eva_{cultivo.lower().replace(' ', '_')}.csv"
    out = proc / file_name
    eva.to_csv(out, index=False)
    print(f"\n✓ {file_name} — {len(eva):,} obs totales")

    # Auditoría
    if "historial" in eva["split"].values:
        audit(eva[eva["split"] == "historial"], "Historial 2007-2018")
    if "target" in eva["split"].values:
        audit(eva[eva["split"] == "target"],    "Target 2019-2024")

    return eva


if __name__ == "__main__":
    main()
