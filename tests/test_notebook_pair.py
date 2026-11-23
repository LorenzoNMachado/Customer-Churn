from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "analise_churn.ipynb"
PYTHON_PAIR = ROOT / "notebooks" / "analise_churn.py"


def _inputs(path: Path) -> list[tuple[str, str]]:
    notebook = jupytext.read(path)
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


def test_notebook_and_python_pair_have_identical_inputs() -> None:
    assert _inputs(NOTEBOOK) == _inputs(PYTHON_PAIR)
