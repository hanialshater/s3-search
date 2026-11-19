"""demo/colab_demo.ipynb (S5-5): valid nbformat, committed without outputs, and every code cell compiles.

The notebook itself is run by hand (Colab, or locally with nbclient); this only keeps it from rotting silently. `%` and `!`
lines (IPython magics) are stripped before compiling. About 0.1 s.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[2] / "demo" / "colab_demo.ipynb"


def _cells():
    return json.loads(NB.read_text())["cells"]


def test_valid_nbformat():
    nbformat = __import__("pytest").importorskip("nbformat")
    nbformat.validate(nbformat.read(str(NB), as_version=4))


def test_committed_without_outputs():
    for c in _cells():
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None, "clear the outputs before committing"


def test_code_cells_compile():
    n = 0
    for i, c in enumerate(_cells()):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        compile(src, f"colab_demo.ipynb cell {i}", "exec")
        n += 1
    assert n >= 10


def test_never_reads_test_judgments_or_deploys():
    text = "".join("".join(c["source"]) for c in _cells())
    for banned in ("_load_test_judgments", "_load_dataset_test_judgments", "[deploy]"):
        assert banned not in text
    assert "development only" in text.lower()
