"""Regenerate all NPY, feature, and XML plots."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)

SCRIPTS = [
    "plotting_npy_pics.py",
    "plotting_npy_graph.py",
    "plot_features_grid.py",
    "plot_xml_footprints.py",
]


def main() -> None:
    for script_name in SCRIPTS:
        script_path = PROJECT_ROOT / script_name
        print("=" * 80)
        print(f"Running {script_name}")
        print("=" * 80)
        result = subprocess.run([str(PYTHON), str(script_path)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    print("All plots regenerated.")


if __name__ == "__main__":
    main()
