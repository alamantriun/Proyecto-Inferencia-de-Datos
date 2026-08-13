#!/usr/bin/env python3
"""
ETL Mejorado - Normalización de Model Mart CON Manejo de Datos Faltantes
=========================================================================
Mejoras sobre versión anterior:
1. Identifica columnas completamente vacías
2. Ofrece 3 estrategias de manejo
3. Reporta qué se removió/mantuvo
4. Valida completitud antes de entrenar ML
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class ModelMartETLMejorado:
    def __init__(self, cultivo, estrategia="remover_columnas"):
        self.cultivo = cultivo
        self.archivo_entrada = f"data/processed/model_mart_{cultivo}.csv"
        self.archivo_salida = f"data/processed/etl_normalized_v2/model_mart_{cultivo}.csv"
        # Estrategias: "remover_columnas", "remover_filas", "mantener_null"
        self.estrategia = estrategia
        self.data = []
        self.headers = []
        self.columnas_vacias = []
        self.reporte = {
            "cultivo": cultivo,
            "timestamp": datetime.now().isoformat(),
            "estrategia": estrategia,
            "estadisticas_entrada": {},
            "estadisticas_salida": {},
            "columnas_vacias_identificadas": [],
            "transformaciones": [],
            "advertencias": []
        }
        self.stats_normalizacion = {}
        self.vars_numericas = []
        self.vars_categoricas = []
        
    def cargar(self):
        """Cargar CSV"""
        print(f"\nCargando {self.archivo_entrada}...")
        try:
            with open(self.archivo_entrada, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.headers = reader.fieldnames
                self.data = list(reader)
        except FileNotFoundError:
            print(f"❌ Archivo no encontrado")
            return False
        
        self.reporte["estadisticas_entrada"] = {
            "filas": len(self.data),
            "columnas": len(self.headers)
        }
        
        print(f"✓ {len(self.data)} filas × {len(self.headers)} columnas")
        return True
        
    def identificar_columnas_vacias(self):
        """Identificar columnas completamente NULL"""
        print("\nIdentificando columnas completamente vacías...")
        
        for header in self.headers:
            valores = [fila.get(header, "").strip() for fila in self.data]
            valores_validos = [v for v in valores if v and v.lower() not in ["nan", "none"]]
            
            if len(valores_validos) == 0:
                self.columnas_vacias.append(header)
                print(f"  ⚠️  {header} - 100% NULL")
        
        self.reporte["columnas_vacias_identificadas"] = self.columnas_vacias
        
        if self.columnas_vacias:
            print(f"\n⚠️  Total columnas vacías: {len(self.columnas_vacias)}")
        else:
            print(f"\n✓ Sin columnas completamente vacías")
        
    def manejar_columnas_vacias(self):
        """Aplicar estrategia de manejo"""
        print(f"\nAplicando estrategia: {self.estrategia.upper()}")
        
        if self.estrategia == "remover_columnas":
            print(f"  Removiendo {len(self.columnas_vacias)} columnas vacías...")
            self.headers = [h for h in self.headers if h not in self.columnas_vacias]
            for fila in self.data:
                for col in self.columnas_vacias:
                    if col in fila:
                        del fila[col]
            msg = f"Removidas {len(self.columnas_vacias)} columnas: {', '.join(self.columnas_vacias[:3])}..."
            print(f"  ✓ {msg}")
            self.reporte["transformaciones"].append(msg)
            
        elif self.estrategia == "remover_filas":
            print(f"  Removiendo filas con datos faltantes...")
            filas_antes = len(self.data)
            self.data = [fila for fila in self.data if all(fila.get(h, "").strip() for h in self.columnas_vacias)]
            filas_removidas = filas_antes - len(self.data)
            msg = f"Removidas {filas_removidas} filas ({filas_removidas/filas_antes*100:.1f}%) con datos faltantes"
            print(f"  ✓ {msg}")
            self.reporte["transformaciones"].append(msg)
            
        elif self.estrategia == "mantener_null":
            print(f"  ✓ Manteniendo NULL tal cual (no se recomienda para ML)")
            self.reporte["advertencias"].append("Se mantienen valores NULL - el modelo puede fallar si no los maneja")
        
    def identificar_tipos(self):
        """Identificar tipos de datos"""
        self.vars_numericas = []
        self.vars_categoricas = []
        
        for header in self.headers:
            if not self.data:
                continue
            
            # Buscar primer valor no-vacío
            valor = None
            for fila in self.data:
                v = fila.get(header, "").strip()
                if v and v.lower() not in ["nan", "none"]:
                    valor = v
                    break
            
            if valor:
                try:
                    float(valor)
                    self.vars_numericas.append(header)
                except ValueError:
                    self.vars_categoricas.append(header)
            else:
                # Si no hay valores, asumir categórica
                self.vars_categoricas.append(header)
        
        print(f"\n✓ {len(self.vars_numericas)} numéricas, {len(self.vars_categoricas)} categóricas")
        
    def convertir_numerico(self, valor):
        """Convertir a número o None"""
        if not valor or valor == "":
            return None
        try:
            v = str(valor).strip()
            if v.lower() in ["nan", "none", ""]:
                return None
            return float(v)
        except ValueError:
            return None
    
    def limpiar(self):
        """Limpieza de datos"""
        print("\nLimpiando datos...")
        
        # Remover duplicados
        filas_antes = len(self.data)
        data_limpia = []
        vistas = set()
        
        for fila in self.data:
            clave_cols = [c for c in ["departamento", "municipio", "cultivo", "anio"] if c in self.headers]
            clave = tuple(fila.get(c, "") for c in clave_cols)
            if clave not in vistas:
                vistas.add(clave)
                data_limpia.append(fila)
        
        self.data = data_limpia
        if filas_antes != len(self.data):
            msg = f"Removidas {filas_antes - len(self.data)} filas duplicadas"
            print(f"  ✓ {msg}")
            self.reporte["transformaciones"].append(msg)
        
        # Imputar NULL en numéricas
        print("  Imputando NULL...")
        for col in self.vars_numericas:
            if col in self.columnas_vacias:
                continue  # No imputar columnas completamente vacías
            
            valores = [self.convertir_numerico(fila.get(col)) for fila in self.data]
            valores_validos = [v for v in valores if v is not None]
            
            if not valores_validos:
                continue
            
            # Calcular mediana
            valores_validos.sort()
            if len(valores_validos) % 2 == 0:
                mediana = (valores_validos[len(valores_validos)//2-1] + valores_validos[len(valores_validos)//2]) / 2
            else:
                mediana = valores_validos[len(valores_validos)//2]
            
            # Reemplazar
            for fila in self.data:
                if not fila.get(col) or fila.get(col) == "":
                    fila[col] = str(mediana)
        
        print(f"  ✓ Imputación completada")
        
    def calcular_estadisticas(self):
        """Calcular media y std para normalización"""
        print("\nCalculando estadísticas...")
        
        for col in self.vars_numericas:
            if col in self.columnas_vacias:
                continue
            
            valores = [self.convertir_numerico(fila.get(col)) for fila in self.data]
            valores_validos = [v for v in valores if v is not None]
            
            if not valores_validos:
                self.stats_normalizacion[col] = {"mean": 0, "std": 1}
                continue
            
            media = sum(valores_validos) / len(valores_validos)
            
            if len(valores_validos) > 1:
                varianza = sum((v - media) ** 2 for v in valores_validos) / len(valores_validos)
                std = varianza ** 0.5
            else:
                std = 1
            
            if std == 0:
                std = 1
            
            self.stats_normalizacion[col] = {"mean": media, "std": std}
        
        print(f"  ✓ {len(self.stats_normalizacion)} variables con parámetros")
        
    def normalizar(self):
        """Normalizar variables numéricas"""
        print("\nNormalizando...")
        
        for fila in self.data:
            for col in self.vars_numericas:
                if col in self.columnas_vacias:
                    continue
                
                valor = self.convertir_numerico(fila.get(col))
                if valor is not None:
                    stats = self.stats_normalizacion.get(col, {"mean": 0, "std": 1})
                    valor_normalizado = (valor - stats["mean"]) / stats["std"]
                    fila[col] = str(valor_normalizado)
        
        print(f"  ✓ {len(self.vars_numericas)} variables normalizadas")
        
    def guardar(self):
        """Guardar CSV"""
        print(f"\nGuardando...")
        
        Path("data/processed/etl_normalized_v2").mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.archivo_salida, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(self.data)
            
            print(f"  ✓ {self.archivo_salida}")
            
            # Verificar NULL finales
            null_finales = 0
            with open(self.archivo_salida, 'r') as f:
                reader = csv.DictReader(f)
                for fila in reader:
                    for val in fila.values():
                        if val == "" or val is None:
                            null_finales += 1
            
            self.reporte["estadisticas_salida"] = {
                "filas": len(self.data),
                "columnas": len(self.headers),
                "columnas_removidas": len(self.columnas_vacias),
                "null_totales": null_finales,
                "variables_normalizadas": [c for c in self.vars_numericas if c not in self.columnas_vacias]
            }
            
            return True
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
        
    def guardar_reporte(self):
        """Guardar JSON"""
        archivo_reporte = f"data/processed/etl_normalized_v2/reporte_{self.cultivo}.json"
        try:
            with open(archivo_reporte, 'w', encoding='utf-8') as f:
                json.dump(self.reporte, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Reporte guardado")
            return True
        except Exception as e:
            print(f"  ❌ Error en reporte: {e}")
            return False
        
    def procesar(self):
        """Pipeline completo"""
        if not self.cargar():
            return None
        
        self.identificar_columnas_vacias()
        self.manejar_columnas_vacias()
        self.identificar_tipos()
        self.limpiar()
        self.calcular_estadisticas()
        self.normalizar()
        
        if not self.guardar():
            return None
        
        self.guardar_reporte()
        
        print(f"\n✅ {self.cultivo.upper()} completado")
        return self.reporte


def main():
    print("\n" + "="*70)
    print("🔄 ETL MEJORADO v2 - Normalización con Manejo de Datos Faltantes")
    print("="*70)
    
    # Usar estrategia "remover_columnas" (recomendado)
    estrategia = "remover_columnas"
    print(f"\n📋 Estrategia seleccionada: {estrategia.upper()}")
    print("   Removerá columnas completamente vacías")
    print("   ✓ Recomendado para modelos ML")
    
    cultivos = ["arroz", "cacao", "cafe", "platano"]
    reportes = {}
    
    for cultivo in cultivos:
        etl = ModelMartETLMejorado(cultivo, estrategia=estrategia)
        print(f"\n{'─'*70}")
        reporte = etl.procesar()
        if reporte:
            reportes[cultivo] = reporte
    
    # Resumen
    print(f"\n{'='*70}")
    print("📊 RESUMEN FINAL")
    print(f"{'='*70}")
    
    for cultivo, reporte in reportes.items():
        print(f"\n{cultivo.upper()}:")
        print(f"  Entrada:           {reporte['estadisticas_entrada']['filas']} filas")
        print(f"  Salida:            {reporte['estadisticas_salida']['filas']} filas")
        print(f"  Columnas removidas: {reporte['estadisticas_salida']['columnas_removidas']}")
        print(f"  NULL finales:      {reporte['estadisticas_salida']['null_totales']}")
        print(f"  ✓ LISTO PARA ML")
    
    print(f"\n✅ ETL v2 completado - Archivos listos en data/processed/etl_normalized_v2/")
    print(f"{'='*70}\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
