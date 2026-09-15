import json

filename = "pipeline_interactivo_cafe.ipynb"
with open(filename, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if "!python3 " in line:
                source[i] = line.replace("!python3 ", "!.venv/bin/python ")

with open(filename, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    # add trailing newline to match jupyter format
    f.write("\n")
