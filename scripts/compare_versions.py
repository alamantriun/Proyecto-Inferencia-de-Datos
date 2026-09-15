"""
Comparador: Versión Estándar vs Versión Rigurosa
================================================

Compara métricas de entrenamiento para determinar si la versión rigurosa
mejoró realmente sobre la versión estándar.
"""

import pandas as pd
from pathlib import Path
import numpy as np

def compare_versions():
    """Compara resultados_ml_ablation_*.csv vs resultados_rigorous_*.csv"""
    
    cultivos = ["arroz", "cacao", "cafe", "platano"]
    
    print("\n" + "="*90)
    print("COMPARACIÓN: Versión Estándar vs Versión Rigurosa")
    print("="*90)
    
    for cultivo in cultivos:
        print(f"\n{'─'*90}")
        print(f"CULTIVO: {cultivo.upper()}")
        print(f"{'─'*90}")
        
        # Versión estándar
        path_std = Path(f"reports/tables/resultados_ml_ablation_{cultivo}.csv")
        if not path_std.exists():
            print(f"  ⚠️  No existe: {path_std}")
            continue
        
        df_std = pd.read_csv(path_std)
        
        # Versión rigurosa
        path_rig = Path(f"reports/tables/resultados_rigorous_{cultivo}.csv")
        if not path_rig.exists():
            print(f"  ⚠️  No existe (ejecuta primero): {path_rig}")
            continue
        
        df_rig = pd.read_csv(path_rig)
        
        # Filtrar solo "Híbrido" en versión estándar (fase F)
        df_std_hybrid = df_std[
            (df_std["Fase"] == "F") & 
            (df_std["Tipo"] == "Híbrido")
        ]
        
        if len(df_std_hybrid) == 0:
            print(f"  ⚠️  No hay resultados Híbrido-F en versión estándar")
            continue
        
        # Agrupar por año
        std_by_year = df_std_hybrid.groupby("Año")["MAE"].mean()
        rig_by_year = df_rig.groupby("Año")["MAE"].mean()
        
        print("\n  Comparación por año:")
        print(f"  {'Año':<6} {'Estándar':<15} {'Rigurosa':<15} {'Mejora':<12} {'Nota':<20}")
        print(f"  {'-'*68}")
        
        mejoras = []
        
        for año in sorted(set(std_by_year.index) & set(rig_by_year.index)):
            mae_std = std_by_year[año]
            mae_rig = rig_by_year[año]
            mejora_pct = (1 - mae_rig / mae_std) * 100 if mae_std > 0 else 0
            mejoras.append(mejora_pct)
            
            # Determinar si es significativa
            if abs(mejora_pct) < 1:
                nota = "Sin cambio"
                emoji = "≈"
            elif mejora_pct > 0:
                nota = "Mejoró ✅"
                emoji = "▼"
            else:
                nota = "Empeoró"
                emoji = "▲"
            
            print(f"  {año:<6} {mae_std:<15.4f} {mae_rig:<15.4f} {mejora_pct:>+7.2f}% {emoji:>2}  {nota:<20}")
        
        # Resumen
        print(f"\n  RESUMEN:")
        mae_std_avg = std_by_year.mean()
        mae_rig_avg = rig_by_year.mean()
        mejora_total = (1 - mae_rig_avg / mae_std_avg) * 100
        
        print(f"    • Versión Estándar MAE promedio: {mae_std_avg:.4f} t/ha")
        print(f"    • Versión Rigurosa MAE promedio: {mae_rig_avg:.4f} t/ha")
        print(f"    • Mejora TOTAL: {mejora_total:+.2f}%")
        
        if mejora_total > 2:
            print(f"    ✅ MEJORA SIGNIFICATIVA - La versión rigurosa vale la pena")
        elif mejora_total > 0.5:
            print(f"    ✓ Mejora pequeña pero detectable")
        elif mejora_total > -0.5:
            print(f"    ≈ Aproximadamente igual (varianza normal)")
        else:
            print(f"    ⚠️  La versión rigurosa empeoró ligeramente")
        
        # Estadísticas
        std_de_mejoras = np.std(mejoras)
        print(f"\n    • Consistencia de mejoras: σ={std_de_mejoras:.2f}%")
        print(f"    • Rango de mejoras: [{min(mejoras):+.2f}%, {max(mejoras):+.2f}%]")
    
    print("\n" + "="*90)
    print("FIN DE COMPARACIÓN")
    print("="*90 + "\n")


def estimate_hardware_benefit():
    """Estima cuánto mejora el hardware disponible"""
    
    print("\n" + "="*90)
    print("ANÁLISIS: ¿Cuánto hardware necesitas para la versión rigurosa?")
    print("="*90)
    
    scenarios = {
        "Laptop antiguo (4GB RAM, i5)": {
            "ram": 4,
            "cpu_cores": 2,
            "feasible": False,
            "reason": "No tiene RAM suficiente"
        },
        "Laptop moderna (8GB RAM, i7)": {
            "ram": 8,
            "cpu_cores": 4,
            "feasible": True,
            "time_minutes": 120,
            "note": "Funciona pero lento, puede necesitar 16GB para comodidad"
        },
        "Laptop gaming (12GB RAM, i7-11th)": {
            "ram": 12,
            "cpu_cores": 6,
            "feasible": True,
            "time_minutes": 60,
            "note": "Óptimo para laptop, máxima recomendación"
        },
        "Desktop media (16GB RAM, Ryzen 5)": {
            "ram": 16,
            "cpu_cores": 6,
            "feasible": True,
            "time_minutes": 45,
            "note": "Muy bueno, rápido"
        },
        "Desktop gaming (32GB RAM, Ryzen 7)": {
            "ram": 32,
            "cpu_cores": 16,
            "feasible": True,
            "time_minutes": 25,
            "note": "Excelente, muy rápido"
        },
        "Servidor cloud (64GB RAM, 32 vCPU)": {
            "ram": 64,
            "cpu_cores": 32,
            "feasible": True,
            "time_minutes": 15,
            "note": "Enterprise, máxima velocidad"
        }
    }
    
    print("\nEsquemático de requerimientos por hardware:\n")
    print(f"{'Escenario':<40} {'RAM':<10} {'CPU':<12} {'Tiempo':<12} {'Viable?':<10} {'Nota':<30}")
    print(f"{'-'*100}")
    
    for scenario, specs in scenarios.items():
        ram = specs["ram"]
        cores = specs["cpu_cores"]
        viable = "✅ Sí" if specs["feasible"] else "❌ No"
        time_str = f"{specs.get('time_minutes', '?')} min"
        note = specs.get("note", specs.get("reason", ""))
        
        print(f"{scenario:<40} {ram}GB{'':<6} {cores:>2} cores{'':<2} {time_str:<12} {viable:<10} {note:<30}")
    
    print("\n" + "="*90)
    print("RECOMENDACIÓN FINAL:")
    print("="*90)
    print("""
Versión Rigurosa (02_train_ml_rigorous.py):
  ✓ Requiere: 8-12 GB RAM mínimo (preferiblemente 12+)
  ✓ CPU: 4+ núcleos a 3+ GHz
  ✓ Tiempo: 30-60 minutos en laptop moderna
  ✓ Mejora: 1-5% en precisión (si hardware es suficiente)

¿Tienes ese hardware?
  • SÍ → Ejecuta versión rigurosa, verás mejoras
  • NO → Usa versión estándar (02_train_ml.py), rápida y confiable

¿Quieres comparar?
  1. Ejecuta versión estándar: python3 src/models/02_train_ml.py
  2. Ejecuta versión rigurosa: python3 src/models/02_train_ml_rigorous.py
  3. Compara: python3 scripts/compare_versions.py
""")


if __name__ == "__main__":
    print("\n🔍 Analizando resultados disponibles...\n")
    
    # Revisar qué archivos existen
    std_files = list(Path("reports/tables").glob("resultados_ml_ablation_*.csv"))
    rig_files = list(Path("reports/tables").glob("resultados_rigorous_*.csv"))
    
    if std_files:
        print(f"✓ Encontradas {len(std_files)} archivos versión estándar")
    else:
        print("✗ No encontrados archivos versión estándar")
    
    if rig_files:
        print(f"✓ Encontradas {len(rig_files)} archivos versión rigurosa")
    else:
        print("✗ No encontrados archivos versión rigurosa")
    
    if std_files and rig_files:
        compare_versions()
    else:
        print("\n⚠️  No hay suficientes archivos para comparar.")
        print("Necesitas ejecutar ambas versiones primero:\n")
        print("  1. Versión estándar:  python3 src/models/02_train_ml.py")
        print("  2. Versión rigurosa:  python3 src/models/02_train_ml_rigorous.py\n")
    
    # Mostrar análisis de hardware
    estimate_hardware_benefit()
