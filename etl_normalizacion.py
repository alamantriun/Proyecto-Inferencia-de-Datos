#!/usr/bin/env python3
"""
ETL Pipeline Simplificado - Normalización de Tablas Model Mart
================================================================
Procesa archivos model_mart_*.csv sin dependencias externas
- Limpieza de datos (NULL, duplicados)
- Normalización de variables numéricas (Z-score)
- Validación de calidad
- Generación de reportes

Solo usa: csv, json, collections (stdlib)
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

class ModelMartETLSimple:
    def __init__(self, cultivo):
        self.cultivo = cultivo
        self.archivo_entrada = f"data/processed/model_mart_{cultivo}.csv"
        self.archivo_salida = f"data/processed/etl_normalized/model_mart_{cultivo}.csv"
        self.data = []
        self.headers = []
        self.reporte = {
            "cultivo": cultivo,
            "timestamp": datetime.now().isoformat(),
            "estadisticas_entrada": {},
            "estadisticas_salida": {},
            "transformaciones": [],
            "advertencias": [],
            "parametros_normalizacion": {}
        }
        self.stats_normalizacion = {}
        
    def cargar(self):
        """Cargar CSV"""
        print(f"\n{'='*60}")
        print(f"PROCESANDO: {self.cultivo.upper()}")
        print(f"{'='*60}")
        print(f"Cargando {self.archivo_entrada}...")
        
        try:
            with open(self.archivo_entrada, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.headers = reader.fieldnames
                self.data = list(reader)
        except FileNotFoundError:
            print(f"❌ ERROR: Archivo no encontrado: {self.archivo_entrada}")
            return False
        
        self.reporte["estadisticas_entrada"] = {
            "filas": len(self.data),
            "columnas": len(self.headers),
            "columnas_lista": self.headers
        }
        
        print(f"✓ Cargado: {len(self.data)} filas × {len(self.headers)} columnas")
        return True
        
    def identificar_tipos(self):
        """Identificar qué columnas son numéricas"""
        print("\nIdentificando tipos de datos...")
        
        self.vars_numericas = []
        self.vars_categoricas = []
        
        if not self.data:
            return
        
        # Revisar primera fila para inferir tipos
        for header in self.headers:
            valor = self.data[0].get(header, "")
            
            if not valor or valor == "":
                # Revisar más filas si está vacío
                es_numerico = False
                for fila in self.data:
                    v = fila.get(header, "").strip()
                    if v and v != "":
                        try:
                            float(v)
                            es_numerico = True
                            break
                        except ValueError:
                            es_numerico = False
                            break
            else:
                try:
                    float(valor)
                    es_numerico = True
                except ValueError:
                    es_numerico = False
            
            if es_numerico:
                self.vars_numericas.append(header)
            else:
                self.vars_categoricas.append(header)
        
        print(f"  Variables numéricas: {len(self.vars_numericas)}")
        print(f"    {', '.join(self.vars_numericas[:5])}...")
        print(f"  Variables categóricas: {len(self.vars_categoricas)}")
        
    def convertir_numerico(self, valor):
        """Convertir string a número, retornando None si no es posible"""
        if valor is None or valor == "":
            return None
        try:
            v = str(valor).strip()
            if v == "" or v.lower() == "nan" or v.lower() == "none":
                return None
            return float(v)
        except (ValueError, AttributeError):
            return None
    
    def limpiar(self):
        """Limpieza básica de datos"""
        print("\nLimpiando datos...")
        
        # 1. Remover duplicados
        filas_antes = len(self.data)
        data_limpia = []
        vistas = set()
        
        for fila in self.data:
            # Crear clave de identificación (departamento, municipio, cultivo, anio)
            clave_cols = [col for col in ["departamento", "municipio", "cultivo", "anio"] if col in self.headers]
            clave = tuple(fila.get(col, "") for col in clave_cols)
            
            if clave not in vistas:
                vistas.add(clave)
                data_limpia.append(fila)
        
        self.data = data_limpia
        filas_despues = len(self.data)
        
        if filas_antes != filas_despues:
            msg = f"  ✓ Removidas {filas_antes - filas_despues} filas duplicadas"
            print(msg)
            self.reporte["transformaciones"].append(msg)
        
        # 2. Imputar NULL en variables numéricas
        print("  ✓ Imputando valores NULL...")
        
        for col in self.vars_numericas:
            valores = [self.convertir_numerico(fila.get(col)) for fila in self.data]
            valores_validos = [v for v in valores if v is not None]
            
            if not valores_validos:
                continue
            
            # Usar mediana o media
            if len(valores_validos) > 0:
                valores_validos.sort()
                if len(valores_validos) % 2 == 0:
                    mediana = (valores_validos[len(valores_validos)//2-1] + valores_validos[len(valores_validos)//2]) / 2
                else:
                    mediana = valores_validos[len(valores_validos)//2]
            else:
                mediana = 0
            
            # Reemplazar NULL
            for fila in self.data:
                if fila.get(col) is None or fila.get(col) == "":
                    fila[col] = str(mediana)
        
        print(f"✓ Limpieza completada")
        
    def calcular_estadisticas(self):
        """Calcular media y desviación estándar para normalización"""
        print("\nCalculando estadísticas para normalización...")
        
        for col in self.vars_numericas:
            valores = [self.convertir_numerico(fila.get(col)) for fila in self.data]
            valores_validos = [v for v in valores if v is not None]
            
            if not valores_validos:
                self.stats_normalizacion[col] = {"mean": 0, "std": 1}
                continue
            
            # Calcular media
            media = sum(valores_validos) / len(valores_validos)
            
            # Calcular desviación estándar
            if len(valores_validos) > 1:
                varianza = sum((v - media) ** 2 for v in valores_validos) / len(valores_validos)
                std = varianza ** 0.5
            else:
                std = 1
            
            # Evitar división por cero
            if std == 0:
                std = 1
            
            self.stats_normalizacion[col] = {"mean": media, "std": std}
            
            if col in self.vars_numericas[:3]:  # Mostrar solo primeras 3
                print(f"  {col}: media={media:.4f}, std={std:.4f}")
        
        # Guardar en reporte
        self.reporte["parametros_normalizacion"] = self.stats_normalizacion
        
    def normalizar(self):
        """Normalizar variables numéricas (Z-score)"""
        print("\nNormalizando variables numéricas...")
        
        for fila in self.data:
            for col in self.vars_numericas:
                valor = self.convertir_numerico(fila.get(col))
                
                if valor is not None:
                    stats = self.stats_normalizacion.get(col, {"mean": 0, "std": 1})
                    valor_normalizado = (valor - stats["mean"]) / stats["std"]
                    fila[col] = str(valor_normalizado)
        
        print(f"✓ {len(self.vars_numericas)} variables normalizadas")
        
    def guardar(self):
        """Guardar archivo procesado"""
        print(f"\nGuardando archivo procesado...")
        
        # Crear directorio
        Path("data/processed/etl_normalized").mkdir(parents=True, exist_ok=True)
        
        # Escribir CSV
        try:
            with open(self.archivo_salida, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(self.data)
            
            print(f"✓ Guardado: {self.archivo_salida}")
            
            self.reporte["estadisticas_salida"] = {
                "filas": len(self.data),
                "columnas": len(self.headers),
                "variables_normalizadas": self.vars_numericas
            }
            
            return True
        except Exception as e:
            print(f"❌ Error al guardar: {e}")
            return False
        
    def guardar_reporte(self):
        """Guardar reporte JSON"""
        archivo_reporte = f"data/processed/etl_normalized/reporte_{self.cultivo}.json"
        
        try:
            with open(archivo_reporte, 'w', encoding='utf-8') as f:
                json.dump(self.reporte, f, indent=2, ensure_ascii=False)
            print(f"✓ Reporte: {archivo_reporte}")
            return True
        except Exception as e:
            print(f"❌ Error al guardar reporte: {e}")
            return False
        
    def procesar(self):
        """Ejecutar pipeline completo"""
        if not self.cargar():
            return None
            
        self.identificar_tipos()
        self.limpiar()
        self.calcular_estadisticas()
        self.normalizar()
        
        if not self.guardar():
            return None
            
        self.guardar_reporte()
        
        print(f"\n✅ PROCESAMIENTO COMPLETADO: {self.cultivo.upper()}")
        print(f"   Entrada: {self.reporte['estadisticas_entrada']['filas']} filas")
        print(f"   Salida: {self.reporte['estadisticas_salida']['filas']} filas")
        
        return self.reporte


def main():
    """Ejecutar ETL para todos los cultivos"""
    print("\n" + "="*60)
    print("🔄 ETL PIPELINE - NORMALIZACIÓN MODEL MART")
    print("="*60)
    
    cultivos = ["arroz", "cacao", "cafe", "platano"]
    reportes = {}
    
    try:
        for cultivo in cultivos:
            etl = ModelMartETLSimple(cultivo)
            reporte = etl.procesar()
            if reporte:
                reportes[cultivo] = reporte
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Resumen consolidado
    print("\n" + "="*60)
    print("📊 RESUMEN CONSOLIDADO")
    print("="*60)
    
    for cultivo, reporte in reportes.items():
        print(f"\n{cultivo.upper()}:")
        print(f"  Entrada:  {reporte['estadisticas_entrada']['filas']} filas")
        print(f"  Salida:   {reporte['estadisticas_salida']['filas']} filas")
        print(f"  Variables normalizadas: {len(reporte['estadisticas_salida']['variables_normalizadas'])}")
    
    # Guardar resumen global
    resumen_global = {
        "pipeline": "Model Mart ETL - Normalización",
        "fecha": datetime.now().isoformat(),
        "cultivos_procesados": list(reportes.keys()),
        "total_cultivos": len(reportes),
        "reportes_individuales": {k: f"reporte_{k}.json" for k in reportes.keys()}
    }
    
    try:
        with open("data/processed/etl_normalized/resumen_global.json", 'w', encoding='utf-8') as f:
            json.dump(resumen_global, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Resumen global guardado")
    except Exception as e:
        print(f"\n⚠️ No se pudo guardar resumen global: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ PIPELINE COMPLETADO")
    print(f"📁 Directorio de salida: data/processed/etl_normalized/")
    print(f"📊 Archivos generados:")
    print(f"   - model_mart_*.csv (4 archivos normalizados)")
    print(f"   - reporte_*.json (4 reportes de transformación)")
    print(f"   - resumen_global.json (resumen consolidado)")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
