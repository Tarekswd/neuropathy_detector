"""Generate per-model hyperparameter reports for tuned and grouped models.

Produces Markdown reports under `reports/hyperparam_reports/` listing:
- the parameter grid used for GridSearchCV (from models.tune_models.PARAM_DISTRIBUTIONS)
- the best parameters found (from the metrics JSON files)

Run from repository root:
    python generate_hyperparam_reports.py
"""
from __future__ import annotations

import json
from pathlib import Path
import textwrap
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import tune_models


OUT_DIR = PROJECT_ROOT / "reports" / "hyperparam_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_metric_files(model_dir: Path):
    return list(model_dir.glob("*_metrics.json"))


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def param_grid_for(name: str, task: str):
    # reuse helper from tune_models
    try:
        return tune_models.get_param_distribution(name, task)
    except Exception:
        # fallback: read from PARAM_DISTRIBUTIONS directly
        pdist = tune_models.PARAM_DISTRIBUTIONS.get(name, {})
        if name == "lr":
            return dict(pdist.get(task, {}))
        return dict(pdist)


def render_report(model_name: str, task: str, model_type: str, metrics_path: Path, data: dict):
    grid = param_grid_for(model_name, task)
    best = data.get("best_params") or data.get("best_params", "N/A")
    tuning_status = data.get("tuning_status") or ("tuned" if "best_params" in data else "unknown")

    out = []
    out.append(f"# Hyperparameter report — {model_type} / {model_name.upper()} / {task}\n")
    out.append(f"**Metrics file:** {metrics_path.relative_to(PROJECT_ROOT)}\n")
    out.append("## GridSearchCV parameter grid\n")
    if grid:
        for k, v in grid.items():
            out.append(f"- **{k}**: {v}")
    else:
        out.append("- (no grid found)")

    out.append("\n## Tuning status\n")
    out.append(f"- {tuning_status}\n")

    out.append("## Best hyperparameters found\n")
    if isinstance(best, dict):
        for k, v in best.items():
            out.append(f"- **{k}**: {v}")
    else:
        out.append(f"- {best}")

    # include cv summary presence
    if data and "cv_metrics_summary" in data:
        out.append("\n## CV summary present: yes\n")
    else:
        out.append("\n## CV summary present: no\n")

    # write file
    filename = f"{model_type}_{model_name}_{task}_hyperparams.md"
    p = OUT_DIR / filename
    p.write_text("\n".join(out), encoding="utf-8")
    return p


def process_directory(model_dir: Path, model_type: str):
    files = find_metric_files(model_dir)
    reports = []
    for f in files:
        data = load_json(f)
        if not data:
            continue
        # try to infer model name and task
        model = data.get("model")
        task = data.get("task")
        if not model or not task:
            # fall back to filename parsing
            parts = f.stem.split("_")
            if len(parts) >= 2:
                model = model or parts[0]
                task = task or parts[1]
            else:
                continue

        # render
        rpt = render_report(model, task, model_type, f, data)
        reports.append(rpt)
    return reports


def main():
    # directories to cover: ungrouped tuned and grouped tuned
    tuned_dir = PROJECT_ROOT / "models" / "tuned"
    grouped_dir = PROJECT_ROOT / "models" / "grouped_tuned"

    print("Generating reports for ungrouped (tuned) models...")
    r1 = process_directory(tuned_dir, "ungrouped")
    print(f"  wrote {len(r1)} reports to {OUT_DIR}")

    print("Generating reports for grouped (grouped_tuned) models...")
    r2 = process_directory(grouped_dir, "grouped")
    print(f"  wrote {len(r2)} reports to {OUT_DIR}")

    total = len(r1) + len(r2)
    print(f"Done — generated {total} hyperparameter reports in: {OUT_DIR}")


if __name__ == "__main__":
    main()
