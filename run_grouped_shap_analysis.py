from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # neuropathy_detector/
PROJECT_ROOT = ROOT.parent                      # project root


def main() -> None:
    print("Running grouped SHAP analysis (binary + multiclass)...")
    cmd = [
        sys.executable,
        str(ROOT / "run_shap_analysis.py"),
        "--analysis-label", "grouped",
        "--use-best-both",
        "--output-dir", str(PROJECT_ROOT / "shap_analysis"),
    ]
    # forward any extra CLI args (e.g. --max-display)
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
