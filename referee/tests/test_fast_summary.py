"""`referee fast`'s last line says PASSED or FAILED and names the failing tests (CI and readers see the last line)."""
import subprocess
import sys
from pathlib import Path


def test_last_line_names_failures(tmp_path):
    t = tmp_path / "test_planted.py"
    t.write_text("def test_ok():\n    pass\n\ndef test_planted_failure():\n    assert False\n")
    code = ("import sys; from referee.__main__ import _Tally; import pytest\n"
            "tally = _Tally(); rc = pytest.main(['-q', '-p', 'no:cacheprovider', sys.argv[1]], plugins=[tally])\n"
            "print(rc, tally.counts, tally.failed)")
    out = subprocess.run([sys.executable, "-c", code, str(t)], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[2]).stdout.strip().splitlines()[-1]
    assert out.startswith("1 ") and "'failed': 1" in out and "'passed': 1" in out and "test_planted_failure" in out
