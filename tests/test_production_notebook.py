from __future__ import annotations

import ast
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def test_current_generator_has_syntactically_valid_code_cells():
    from scripts.create_mcc_kappa_notebook import build_notebook

    notebook = build_notebook()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            try:
                ast.parse(cell.source)
            except SyntaxError as error:
                raise AssertionError(f"Celda de código {index} inválida: {error}") from error


def test_original_backup_and_replacement_exist():
    assert (ROOT / "pipeline_interactivo_cafe_original.ipynb").is_file()
    assert (ROOT / "pipeline_interactivo_cafe.ipynb").is_file()


def test_replaced_notebook_is_executed_without_error_outputs():
    notebook = nbformat.read(ROOT / "pipeline_interactivo_cafe.ipynb", as_version=4)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]

    assert errors == []
    assert all(
        cell.get("execution_count") is not None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
