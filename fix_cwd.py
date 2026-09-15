import json

filename = "pipeline_interactivo_cafe.ipynb"
with open(filename, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if "# Cargar configuración del proyecto" in line:
                # Insert the chdir command right before this
                source.insert(i, "os.chdir('/home/william/Documentos/proyecto IngDatos')\n")
                break
        cell["source"] = source

with open(filename, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
