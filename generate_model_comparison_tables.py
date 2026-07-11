"""
Generate 4 comparison tables (PNG) for model performance metrics:
  1. Ungrouped — Binary
  2. Ungrouped — Multiclass
  3. Grouped   — Binary
  4. Grouped   — Multiclass

Each table shows 5 algorithms with CV mean ± std for every metric.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALGORITHMS = ["svm", "rf", "lr", "xgb", "catboost"]
TASKS = ["binary", "multiclass"]

UNGROUPED_DIR = PROJECT_ROOT / "models" / "tuned"
GROUPED_DIR = PROJECT_ROOT / "models" / "grouped_tuned"

OUTPUT_DIR = PROJECT_ROOT / "model_comparison_tables"

METRICS = ["balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1_score", "auc"]
METRIC_LABELS = {
    "balanced_accuracy": "Bal. Acc.",
    "accuracy": "Accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1_score": "F1",
    "auc": "AUC",
}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None


def _load_metrics(model_dir: Path, algo: str, task: str) -> dict | None:
    candidates = [
        model_dir / f"{algo}_{task}_grouped_metrics.json",
        model_dir / f"{algo}_{task}_metrics.json",
        model_dir / f"{algo}_metrics.json",
    ]
    for p in candidates:
        data = _load_json(p)
        if data is not None:
            return data
    print(f"  [WARN] No metrics file for {model_dir.name}/{algo}/{task}")
    return None


def _extract_mean_std(data: dict, metric: str) -> str:
    """
    Extract mean ± std from cv_metrics_summary.

    Ungrouped tuned  → cv_metrics_summary[metric] = {"mean": x, "std": y}
    Grouped tuned    → cv_metrics_summary has flat keys: metric_mean / metric_std
                       OR nested fold_metrics list from subject_cross_validation
    """
    cv = data.get("cv_metrics_summary", {})
    if not cv:
        return "N/A"

    # --- nested dict format: {"mean": x, "std": y} ---
    if metric in cv and isinstance(cv[metric], dict):
        mean = cv[metric].get("mean")
        std = cv[metric].get("std", 0.0) or 0.0
        if mean is not None:
            return f"{mean:.3f} ± {std:.3f}"

    # --- flat key format: metric_mean / metric_std ---
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    if mean_key in cv:
        mean = cv[mean_key]
        std = cv.get(std_key, 0.0) or 0.0
        if mean is not None:
            return f"{mean:.3f} ± {std:.3f}"

    # --- sensitivity / recall alias ---
    if metric == "sensitivity":
        alt = _extract_mean_std(data, "recall")
        return alt if alt != "N/A" else "N/A"
    if metric == "recall":
        return _extract_mean_std(data, "sensitivity")

    # --- fallback: compute from fold_metrics list ---
    fold_metrics = cv.get("fold_metrics", [])
    if fold_metrics:
        import numpy as np
        key = "sensitivity" if metric == "recall" else metric
        vals = [f.get(key) for f in fold_metrics if f.get(key) is not None]
        if vals:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            return f"{mean:.3f} ± {std:.3f}"

    return "N/A"


def build_table(model_dir: Path, task: str) -> pd.DataFrame:
    rows = []
    for algo in ALGORITHMS:
        data = _load_metrics(model_dir, algo, task)
        row = {"Algorithm": algo.upper()}
        for metric in METRICS:
            label = METRIC_LABELS[metric]
            row[label] = _extract_mean_std(data, metric) if data else "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def render_table(df: pd.DataFrame, title: str, out_path: Path) -> None:
    n_cols = len(df.columns)
    n_rows = len(df)

    fig_w = max(14, n_cols * 1.6)
    fig_h = max(3, n_rows * 0.65 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_widths = [0.10] + [0.13] * (n_cols - 1)

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.4)

    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white")

    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            tbl[(i, j)].set_facecolor("#F2F2F2" if i % 2 else "#E7E6E6")

    plt.title(title, fontsize=13, fontweight="bold", pad=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("GENERATING 4 MODEL COMPARISON TABLES")
    print("=" * 70)

    configs = [
        (UNGROUPED_DIR, "binary",     "Ungrouped — Binary (CV mean ± std)",     "comparison_ungrouped_binary.png"),
        (UNGROUPED_DIR, "multiclass", "Ungrouped — Multiclass (CV mean ± std)", "comparison_ungrouped_multiclass.png"),
        (GROUPED_DIR,   "binary",     "Grouped — Binary (CV mean ± std)",       "comparison_grouped_binary.png"),
        (GROUPED_DIR,   "multiclass", "Grouped — Multiclass (CV mean ± std)",   "comparison_grouped_multiclass.png"),
    ]

    for model_dir, task, title, filename in configs:
        print(f"\n[{title}]")
        df = build_table(model_dir, task)
        if df.empty:
            print("  [WARN] No data — skipping.")
            continue
        render_table(df, title, OUTPUT_DIR / filename)

    print("\n" + "=" * 70)
    print(f"Done. Tables saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
