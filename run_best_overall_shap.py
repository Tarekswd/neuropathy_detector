"""run_best_overall_shap.py
Run SHAP analysis for the best overall models (binary and multiclass) as determined
by the cross-comparison report output of compare_best_models.py.
For each winner (binary / multiclass):
- Loads the winning model, features, and corresponding train/test data.
- Generates a Beeswarm plot displaying the top 20 features.
- Generates a horizontal Bar plot displaying the top 20 features in descending importance order.
- Saves the outputs in shap_analysis/best_overall/{task}/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
REPORT_PATH = PROJECT_ROOT / "model_comparison_tables" / "cross_comparison_report.json"
SHAP_OUT_DIR = PROJECT_ROOT / "shap_analysis" / "best_overall"

# ─────────────────────────────────────────────────────────────────────────────
# SHAP explanation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_explanation_data(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if hasattr(pipeline, "named_steps") and len(pipeline.named_steps) > 1:
        try:
            transformed = pipeline[:-1].transform(X)
            if isinstance(transformed, pd.DataFrame):
                return transformed.to_numpy(), transformed.columns.tolist()
            return transformed, [f"f{i}" for i in range(transformed.shape[1])]
        except Exception:
            pass
    X_values = X.to_numpy(dtype=float)
    return X_values, X.columns.tolist()


def _build_explainer(pipeline, X_train: np.ndarray, feature_names: list[str]):
    model = pipeline.named_steps.get("model", pipeline)
    if hasattr(model, "coef_"):
        try:
            return shap.LinearExplainer(model, X_train), feature_names
        except Exception:
            pass
    if hasattr(model, "feature_importances_"):
        try:
            return shap.TreeExplainer(model), feature_names
        except Exception:
            pass
    if hasattr(pipeline, "predict_proba"):
        predict_fn = lambda values: pipeline.predict_proba(values)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        predict_fn = lambda values: pipeline.decision_function(values)
    else:
        predict_fn = lambda values: pipeline.predict(values)
    return shap.Explainer(predict_fn, X_train), feature_names


def run_shap_for_winner(winner: dict, output_dir: Path, max_display: int = 20) -> None:
    print(f"\n{'─'*60}")
    print(f"  SHAP analysis for winner: {winner['label']} ({winner['task'].upper()})")
    print(f"  Joblib Path: {winner['joblib_path']}")
    print(f"  Output Dir:  {output_dir}")
    print(f"{'─'*60}")

    joblib_path = Path(winner["joblib_path"])
    if not joblib_path.exists():
        # Try relative paths if absolute path stored doesn't resolve directly
        relative_candidate = PROJECT_ROOT / Path(winner["joblib_path"]).relative_to(Path(winner["joblib_path"]).anchor)
        if relative_candidate.exists():
            joblib_path = relative_candidate
        else:
            # Check relative to project root directly if windows paths mismatched
            parts = Path(winner["joblib_path"]).parts
            for i in range(len(parts)):
                sub_path = PROJECT_ROOT / Path(*parts[i:])
                if sub_path.exists():
                    joblib_path = sub_path
                    break

    if not joblib_path.exists():
        print(f"  [error] joblib model file not found: {winner['joblib_path']}")
        return

    # Load trained model
    artifact = joblib.load(joblib_path)
    pipeline = artifact["model"]
    features = artifact.get("features", [])
    task = winner["task"]
    label = winner["label"]

    # Load dataset
    from models.training_utils import load_train_test_datasets, load_full_dataset
    train_data_path = Path(winner["train_data"])
    test_data_path = Path(winner["test_data"])

    # Attempt to resolve train/test paths relative to root if absolute paths differ
    if not train_data_path.exists() or "patient_features_train.csv" in train_data_path.name:
        if winner["source"] == "grouped":
            train_data_path = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_train.csv"
        else:
            train_data_path = PROJECT_ROOT / "ml_features" / "npy_features_train.csv"
    if not test_data_path.exists() or "patient_features_test.csv" in test_data_path.name:
        if winner["source"] == "grouped":
            test_data_path = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_test.csv"
        else:
            test_data_path = PROJECT_ROOT / "ml_features" / "npy_features_test.csv"

    try:
        X_train, X_test, _, _, _, _, feature_cols = load_train_test_datasets(train_data_path, test_data_path, task)
    except RuntimeError:
        X_train, _, _, feature_cols = load_full_dataset([train_data_path], task)
        X_test, _, _, _ = load_full_dataset([test_data_path], task)

    # Use the features from the artifact if defined
    if features is not None and len(features) > 0:
        # Convert pd.Index or numpy array to a standard list if necessary
        if hasattr(features, "tolist"):
            features_list = features.tolist()
        else:
            features_list = list(features)
        feature_cols = [f for f in features_list if f in X_train.columns]

    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    X_train_proc, train_feature_names = _prepare_explanation_data(pipeline, X_train)
    X_test_proc, test_feature_names = _prepare_explanation_data(pipeline, X_test)
    if train_feature_names != test_feature_names:
        train_feature_names = test_feature_names

    # Build SHAP explainer and compute SHAP values
    explainer, _ = _build_explainer(pipeline, X_train_proc, train_feature_names)
    shap_values = explainer(X_test_proc)

    if isinstance(shap_values, list):
        # Multi-class or multi-output list of SHAP values
        if len(shap_values) > 1:
            # Default to second class if binary classification loaded as multi-class output list
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)

    # Handle multidimensional array shapes
    if values.ndim == 3:
        # If shape is [samples, features, classes], pick the positive class or first active class
        values = values[:, :, 1] if values.shape[2] >= 2 else values[:, :, 0]
    if values.ndim != 2:
        values = values.reshape(values.shape[0], -1)
    if values.shape[1] != len(train_feature_names):
        values = values[:, :len(train_feature_names)]

    # Compute feature importance based on mean absolute SHAP value
    importance_df = (
        pd.DataFrame({"feature": train_feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    title_suffix = f"{label} ({task})"

    # ── 1. Beeswarm Plot (Top 20 Features) ───────────────────────────────────
    plt.figure()
    shap.summary_plot(
        values,
        X_test_proc,
        feature_names=train_feature_names,
        max_display=max_display,
        show=False
    )
    fig_beeswarm = plt.gcf()
    fig_beeswarm.suptitle(f"SHAP beeswarm (Top {max_display}) — {title_suffix}", fontsize=11, y=1.02)
    beeswarm_path = output_dir / "shap_beeswarm_top20.png"
    fig_beeswarm.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close(fig_beeswarm)
    print(f"  Saved beeswarm plot -> {beeswarm_path}")

    # ── 2. Bar Plot (Top 20 Features) ────────────────────────────────────────
    # Extract top max_display features
    top_importance = importance_df.head(max_display).copy()
    # Sort ascending for horizontal bar chart display order
    top_importance_asc = top_importance.sort_values("mean_abs_shap", ascending=True)

    bar_fig, bar_ax = plt.subplots(figsize=(10, max(4, 0.28 * len(top_importance_asc))))
    bar_ax.barh(top_importance_asc["feature"], top_importance_asc["mean_abs_shap"], color="#3b82f6")
    bar_ax.set_xlabel("Mean |SHAP value|")
    bar_ax.set_title(f"Feature importance (mean |SHAP| Top {max_display}) — {title_suffix}")
    bar_ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    bar_path = output_dir / "shap_bar_top20.png"
    bar_fig.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close(bar_fig)
    print(f"  Saved bar plot      -> {bar_path}")

    # Save summary metadata JSON
    summary = {
        "winner_label": label,
        "winner_source": winner["source"],
        "winner_model": winner["model"],
        "task": task,
        "top_features": importance_df.head(max_display).to_dict(orient="records"),
    }
    (output_dir / "shap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    if not REPORT_PATH.exists():
        print(f"Error: Comparison report not found at {REPORT_PATH}.")
        print("Please run `python compare_best_models.py` first to generate the report.")
        sys.exit(1)

    with REPORT_PATH.open(encoding="utf-8") as fh:
        report_data = json.load(fh)

    for task in ["binary", "multiclass"]:
        if task not in report_data or "winner" not in report_data[task]:
            print(f"Skipping {task} as it's not present in the comparison report.")
            continue

        winner = report_data[task]["winner"]
        output_dir = SHAP_OUT_DIR / task
        try:
            run_shap_for_winner(winner, output_dir, max_display=20)
        except Exception as exc:
            import traceback
            print(f"Error running SHAP for {task}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
