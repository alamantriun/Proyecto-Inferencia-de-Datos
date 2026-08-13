#!/bin/bash
# run_multicrop.sh
# Super Orquestador para ejecutar el pipeline predictivo en múltiples cultivos

# Fix Error #3: Guardar una copia del config original ANTES de modificarlo
# y restaurarlo al final para no destruir la estructura del YAML

CROPS=("cacao" "cafe" "platano" "arroz")
CONFIG="config/config.yaml"
CONFIG_BACKUP="config/config.yaml.bak"

echo "=========================================================="
echo "🌱 INICIANDO PIPELINE MULTI-CULTIVO AGRO-RANK 🌱"
echo "Cultivos programados: ${CROPS[*]}"
echo "=========================================================="

# Guardar backup del config original (preserva comentarios y formato)
cp "$CONFIG" "$CONFIG_BACKUP"

for CROP in "${CROPS[@]}"
do
    echo ""
    echo "=========================================================="
    echo "🚀 PROCESANDO: $CROP"
    echo "=========================================================="
    
    # Fix Error #3: En vez de reescribir todo el YAML con yaml.dump(),
    # usamos sed para cambiar SOLO la línea del cultivo_mvp.
    # Esto preserva comentarios, formato y orden del archivo.
    sed -i "s/cultivo_mvp: .*/cultivo_mvp: \"$CROP\"/" "$CONFIG"
    
    # Ejecutar el pipeline completo para este cultivo
    bash run_pipeline.sh
    
    echo "✅ Pipeline completado para: $CROP"
    echo "Resultados disponibles en: reports/tables/ y reports/figures/"
done

# Fix Error #3: Restaurar el config original al finalizar
mv "$CONFIG_BACKUP" "$CONFIG"
echo ""
echo "✓ config.yaml restaurado a su estado original"

echo ""
echo "=========================================================="
echo "🎉 TODOS LOS CULTIVOS HAN SIDO PROCESADOS EXITOSAMENTE 🎉"
echo "=========================================================="
