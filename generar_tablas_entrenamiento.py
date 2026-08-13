#!/usr/bin/env python3
"""
Genera un documento HTML consolidado con todas las tablas de entrenamiento
de los tres modelos (Arroz, Cacao, Café, Plátano).
"""

import csv
from pathlib import Path
from html import escape

def leer_csv(ruta):
    """Lee un archivo CSV y retorna headers y data"""
    with open(ruta, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = list(reader)
    return headers, data

def generar_html():
    """Genera el documento HTML con todas las tablas"""
    
    productos = {
        'arroz': 'Arroz',
        'cacao': 'Cacao',
        'cafe': 'Café',
        'platano': 'Plátano'
    }
    
    # Descripción de las secciones de variables
    descripciones = {
        'Identificadores': [
            'departamento', 'municipio', 'cultivo', 'anio', 'fuente_eva', 'split'
        ],
        'Producción e Historia (Panel)': [
            'area_sembrada_ha', 'area_cosechada_ha', 'produccion_t', 'rendimiento_t_ha',
            'rendimiento_lag_1', 'rendimiento_lag_2', 'rendimiento_lag_3',
            'produccion_lag_1', 'area_cosechada_lag_1', 'area_sembrada_lag_1',
            'media_rendimiento_3y', 'variabilidad_rendimiento_3y', 'tendencia_rendimiento_3y',
            'es_outlier_rendimiento', 'dato_copiado', 'inconsistencia_rend', 'score_confiabilidad'
        ],
        'Clima (IDEAM)': [
            'precipitacion_acumulada_mm', 'dias_lluvia', 'intensidad_max_diaria_mm',
            'precipitacion_mediana_diaria', 'precip_Q3_mm', 'ratio_concentracion_lluvia',
            'cv_precipitacion_mensual', 'max_dias_secos_consecutivos'
        ],
        'Suelos (AGROSAVIA)': [
            'ph_media', 'num_muestras_suelo', 'ph_variabilidad',
            'materia_organica_pct_media', 'materia_organica_pct_variabilidad',
            'fosforo_ppm_media', 'fosforo_ppm_variabilidad',
            'calcio_meq_media', 'calcio_meq_variabilidad',
            'magnesio_meq_media', 'magnesio_meq_variabilidad',
            'potasio_meq_media', 'potasio_meq_variabilidad',
            'salinidad_ds_m_media', 'salinidad_ds_m_variabilidad'
        ],
        'Tendencias de Suelo': [
            'tendencia_ph', 'tendencia_materia_organica_pct', 'tendencia_fosforo_ppm',
            'tendencia_calcio_meq', 'tendencia_magnesio_meq', 'tendencia_potasio_meq',
            'tendencia_salinidad_ds_m'
        ],
        'Inversión (FINAGRO)': [
            'credito_total', 'colocacion_total', 'num_operaciones_credito',
            'credito_promedio_operacion', 'log_credito_total'
        ],
        'Precio Internacional (FRED)': [
            'precio_internacional_usd', 'precio_internacional_max', 'precio_internacional_min',
            'volatilidad_precio', 'cambio_precio_pct', 'rango_precio_ratio'
        ]
    }
    
    html_content = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tablas de Entrenamiento - Modelos de Rendimiento Agrícola</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .producto-section {
            margin-bottom: 50px;
            page-break-inside: avoid;
        }
        
        .producto-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .producto-stats {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .variable-group {
            background: #f8f9fa;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
            border-radius: 4px;
        }
        
        .group-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
        }
        
        .group-title::before {
            content: "▸";
            margin-right: 10px;
            font-size: 1.4em;
        }
        
        .variables-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        
        .variable-item {
            background: white;
            padding: 12px;
            border-radius: 4px;
            border-left: 3px solid #764ba2;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            word-break: break-word;
        }
        
        .table-preview {
            margin-top: 30px;
            overflow-x: auto;
        }
        
        .table-preview h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 0.85em;
        }
        
        thead {
            background: #667eea;
            color: white;
            font-weight: bold;
        }
        
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        tbody tr:hover {
            background: #f5f5f5;
        }
        
        tbody tr:nth-child(even) {
            background: #f9f9f9;
        }
        
        .empty-data {
            text-align: center;
            color: #999;
            padding: 20px;
            font-style: italic;
        }
        
        .footer {
            background: #f0f0f0;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        
        .badge {
            display: inline-block;
            background: #764ba2;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        
        @media print {
            body {
                background: white;
                padding: 0;
            }
            .container {
                box-shadow: none;
                border-radius: 0;
            }
            .producto-section {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Tablas de Entrenamiento</h1>
            <p>Modelos de Predicción de Rendimiento Agrícola</p>
            <p style="margin-top: 10px; font-size: 0.95em;">Arroz • Cacao • Café • Plátano</p>
        </div>
        
        <div class="content">
"""
    
    # Procesar cada producto
    for prod_key, prod_nombre in productos.items():
        ruta_archivo = f"data/processed/model_mart_{prod_key}.csv"
        
        try:
            headers, data = leer_csv(ruta_archivo)
            num_filas = len(data)
            num_cols = len(headers)
            
            html_content += f"""
            <div class="producto-section">
                <div class="producto-title">
                    <span>🌾 {prod_nombre}</span>
                    <span class="producto-stats">{num_filas} filas × {num_cols} variables</span>
                </div>
"""
            
            # Agrupar variables por categoría
            for grupo_nombre, variables in descripciones.items():
                # Filtrar variables que existen en los headers
                vars_presentes = [v for v in variables if v in headers]
                
                if vars_presentes:
                    html_content += f"""
                <div class="variable-group">
                    <div class="group-title">{grupo_nombre}</div>
                    <div class="variables-list">
"""
                    for var in vars_presentes:
                        html_content += f'                        <div class="variable-item">{escape(var)}</div>\n'
                    
                    html_content += """
                    </div>
                </div>
"""
            
            # Mostrar previsualizaciónde datos (primeras 5 filas)
            html_content += f"""
                <div class="table-preview">
                    <h3>📋 Primeras 5 filas de datos</h3>
                    <table>
                        <thead>
                            <tr>
"""
            
            # Headers
            for header in headers[:15]:  # Limitar a primeras 15 columnas para mejor visualización
                html_content += f"                                <th>{escape(header)}</th>\n"
            
            if len(headers) > 15:
                html_content += f'                                <th style="color: #aaa;">+{len(headers)-15} más</th>\n'
            
            html_content += """
                            </tr>
                        </thead>
                        <tbody>
"""
            
            # Datos (primeras 5 filas)
            for fila in data[:5]:
                html_content += "                            <tr>\n"
                for i, valor in enumerate(fila[:15]):
                    valor_display = escape(str(valor)[:50]) if valor else "-"
                    html_content += f"                                <td>{valor_display}</td>\n"
                if len(fila) > 15:
                    html_content += f'                                <td style="color: #aaa;">...</td>\n'
                html_content += "                            </tr>\n"
            
            html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
"""
        
        except FileNotFoundError:
            html_content += f"""
            <div class="producto-section">
                <div class="producto-title">
                    <span>🌾 {prod_nombre}</span>
                    <span class="producto-stats">❌ Archivo no encontrado</span>
                </div>
                <div class="empty-data">El archivo model_mart_{prod_key}.csv no se encontró</div>
            </div>
"""
    
    html_content += """
        </div>
        
        <div class="footer">
            <p>Documento generado automáticamente desde tablas model_mart_*.csv</p>
            <p>Proyecto: Inferencia de Datos - Modelado de Rendimiento Agrícola</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html_content

# Generar el archivo
html = generar_html()
with open('reports/tablas_entrenamiento/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("✅ Archivo HTML generado: reports/tablas_entrenamiento/index.html")

# También copiar los CSVs
import shutil
for prod in ['arroz', 'cacao', 'cafe', 'platano']:
    src = f'data/processed/model_mart_{prod}.csv'
    dst = f'reports/tablas_entrenamiento/model_mart_{prod}.csv'
    if Path(src).exists():
        shutil.copy(src, dst)
        print(f"✅ Copiado: {dst}")
