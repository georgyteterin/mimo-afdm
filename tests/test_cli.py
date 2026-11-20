import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("module", ["runners.ber_vs_ebn0", "runners.ber_vs_ebn0_parallel", "plotters.plot_ber"])
def test_help_works(module):
    completed = subprocess.run([sys.executable, "-m", module, "--help"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
