from pathlib import Path

import nbformat
from nbformat.validator import validate


NOTEBOOK_PATH = Path("mandelbrot.ipynb")


def test_notebook_is_valid_nbformat():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    validate(notebook)


def test_notebook_uses_shared_core_module():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "from mandelbrot.core import" in source
    assert "def mandelbrot_set" not in source
