"""
Paso 05: Extracción UPRA (Aptitud Cacao)
========================================
Extrae datos de la zonificación de aptitud para cultivo de cacao comercial.
Fuente: https://www.datos.gov.co/resource/jdjx-qer4.json

Regla Leakage: UPRA es una variable estructural. La usamos como context features.
"""

import sys
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.soda_client import SodaClient

def extract_upra_aptitud(upra_id: str, cultivo: str) -> pd.DataFrame:
    print(f"\n=== UPRA Aptitud {cultivo.upper()} ===")
    client = SodaClient(upra_id)
    
    # Podemos hacer un query agregado con SODA para no bajar los polígonos
    select = "municipio,aptitud,sum(area_ha) as area_total"
    group = "municipio,aptitud"
    
    df = client.extract_all(
        select=select,
        group=group,
        order="municipio", 
        batch=50000
    )
    
    if df.empty:
        print("⚠ Sin datos")
        return df
        
    df["area_total"] = pd.to_numeric(df["area_total"], errors="coerce")
    
    # Estandarizar string municipio
    df["municipio"] = (
        df["municipio"].astype(str).str.strip().str.upper()
        .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
    )
    
    # Pivotar para que cada municipio sea una fila y las aptitudes sean columnas
    pivot = df.pivot_table(
        index="municipio",
        columns="aptitud",
        values="area_total",
        aggfunc="sum",
        fill_value=0
    )
    
    # Limpiar nombres de columnas
    pivot.columns = [
        "aptitud_" + c.lower().replace(" ", "_").replace("í", "i").replace("ó", "o")
        for c in pivot.columns
    ]
    
    # Calcular porcentajes relativos
    total_area = pivot.sum(axis=1)
    for col in pivot.columns:
        pivot[f"{col}_pct"] = (pivot[col] / total_area).fillna(0)
        
    res = pivot.reset_index()
    print(f"\nPanel UPRA: {len(res):,} municipios con perfil de aptitud.")
    return res


def main():
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    cultivo = config["project"].get("cultivo_mvp", "cacao")
    
    # Obtener el ID de UPRA de config
    upra_info = config["fuentes"].get("upra_aptitud", {}).get(cultivo)
    if not upra_info or not upra_info.get("id_dataset"):
        print(f"⚠ No hay un ID de UPRA configurado para el cultivo: {cultivo}. Saltando extracción.")
        return

    upra_id = upra_info["id_dataset"]

    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    df = extract_upra_aptitud(upra_id, cultivo)
    if not df.empty:
        file_name = f"upra_aptitud_{cultivo.lower().replace(' ', '_')}.csv"
        df.to_csv(proc_dir / file_name, index=False)
        print(f"✓ Guardado: {proc_dir / file_name}")


if __name__ == "__main__":
    main()
