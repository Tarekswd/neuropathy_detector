from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import sys

# for_github/ is the script dir; project root is one level up (contains models/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training_utils import DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, load_train_test_datasets, load_full_dataset

DEFAULT_BEST_MODEL_PATH = _PROJECT_ROOT / "best model" / "best_model.joblib"
DEFAULT_BEST_MODEL_DIR = _PROJECT_ROOT / "best model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis for a trained footprint classification pipeline.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_BEST_MODEL_PATH)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--grouped-train-data", type=Path,
        default=_PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_train.csv",
        help="Grouped train CSV (used when --analysis-label grouped).")
    parser.add_argument("--grouped-test-data", type=Path,
        default=_PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_test.csv",
        help="Grouped test CSV (used when --analysis-label grouped).")
    parser.add_argument("--task", type=str, default="binary")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "shap_analysis")
    parser.add_argument("--max-display", type=int, default=15)
    parser.add_argument("--best-model-dir", type=Path, default=DEFAULT_BEST_MODEL_DIR,
        help="Directory containing best_model_binary.joblib and best_model_multiclass.joblib.")
    parser.add_argument(
        "--grouped-model-dir",
        type=Path,
        default=_PROJECT_ROOT / "best model" / "grouped",
        help="Directory containing grouped best_model_binary.joblib and best_model_multiclass.joblib.",
    )
    parser.add_argument(
        "--analysis-label",
        type=str,
        default="ungrouped",
        choices=("ungrouped", "grouped"),
        help="Label used for the output folder name: ungrouped_shap_analysis or grouped_shap_analysis.",
    )
    parser.add_argument(
        "--use-best-both",
        "--use_best_both",
        dest="use_best_both",
        action="store_true",
        help="Run SHAP on both best binary and best multiclass saved models (looks in the selected best-model-dir).",
    )
    return parser.parse_args()


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

    # Linear models (LR, LinearSVC) — use LinearExplainer
    if hasattr(model, "coef_"):
        try:
            return shap.LinearExplainer(model, X_train), feature_names
        except Exception:
            pass

    # Tree-based models (RF, XGB, CatBoost) — use TreeExplainer
    if hasattr(model, "feature_importances_"):
        try:
            return shap.TreeExplainer(model), feature_names
        except Exception:
            pass

    # Generic fallback via predict_proba / decision_function / predict
    if hasattr(pipeline, "predict_proba"):
        predict_fn = lambda values: pipeline.predict_proba(values)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        predict_fn = lambda values: pipeline.decision_function(values)
    else:
        predict_fn = lambda values: pipeline.predict(values)

    return shap.Explainer(predict_fn, X_train), feature_names


def save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _resolve_model_name(model_path: Path, artifact: Any) -> str:
    for key in ("model_name", "name", "model_type"):
        value = artifact.get(key) if isinstance(artifact, dict) else None
        if value:
            return str(value)
    return model_path.stem


def run_analysis(model_path: Path, train_data: Path, test_data: Path, task: str, output_dir: Path, max_display: int, out_png_name: str | None = None) -> None:
    if not model_path.exists():
        fallback = _PROJECT_ROOT / "models" / "tuned" / "lr_binary_tuned.joblib"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)
    pipeline = artifact["model"]
    features = artifact.get("features", [])
    model_name = _resolve_model_name(model_path, artifact)

    print(f"Model: {model_name}")
    print(f"Model path: {model_path}")

    try:
        X_train, X_test, y_train, y_test, _, _, feature_cols = load_train_test_datasets(train_data, test_data, task)
    except RuntimeError:
        # grouped data may have subject overlap between train/test files — load combined
        X_train, _, _, feature_cols = load_full_dataset([train_data], task)
        X_test, _, _, _ = load_full_dataset([test_data], task)
    if features:
        feature_cols = [f for f in features if f in X_train.columns]
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    X_train_proc, train_feature_names = _prepare_explanation_data(pipeline, X_train)
    X_test_proc, test_feature_names = _prepare_explanation_data(pipeline, X_test)
    if train_feature_names != test_feature_names:
        train_feature_names = test_feature_names

    explainer, _ = _build_explainer(pipeline, X_train_proc, train_feature_names)
    if task == "binary":
        shap_values = explainer(X_test_proc)
    else:
        shap_values = explainer(X_test_proc)

    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[2] >= 2 else values[:, :, 0]
    if values.ndim != 2:
        values = values.reshape(values.shape[0], -1)

    if values.shape[1] != len(train_feature_names):
        values = values[:, : len(train_feature_names)]

    importance_df = pd.DataFrame(
        {
            "feature": train_feature_names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "shap_feature_importance.csv"
    importance_df.to_csv(summary_csv, index=False)

    print("Top SHAP features:")
    print(importance_df.head(10).to_string(index=False))

    # Beeswarm plot — shap.summary_plot manages its own figure
    shap.summary_plot(values, X_test_proc, feature_names=train_feature_names, max_display=max_display, show=False)
    fig = plt.gcf()
    if out_png_name is None:
        out_png = output_dir / "shap_summary_beeswarm.png"
    else:
        out_png = output_dir / out_png_name
    save_plot(fig, out_png)

    # Produce a bar plot of mean absolute SHAP values for all features
    try:
        bar_fig, bar_ax = plt.subplots(figsize=(10, max(4, 0.25 * len(train_feature_names))))
        # ensure we plot in descending importance order
        imp_sorted = importance_df.sort_values("mean_abs_shap", ascending=True)
        bar_ax.barh(imp_sorted["feature"], imp_sorted["mean_abs_shap"], color="tab:blue")
        bar_ax.set_xlabel("Mean |SHAP value|")
        bar_ax.set_title(f"{model_name} — Feature importance (mean |SHAP|) — all features")
        bar_ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        bar_out = output_dir / "shap_summary_bar.png"
        save_plot(bar_fig, bar_out)
        print(f"Wrote SHAP bar plot -> {bar_out}")
    except Exception as e:
        print(f"Failed to produce SHAP bar plot: {e}")

    metrics_path = output_dir / "shap_summary.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_path": str(model_path),
                "task": task,
                "feature_count": len(train_feature_names),
                "top_features": importance_df.head(10).to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote SHAP summary CSV -> {summary_csv}")
    print(f"Wrote SHAP beeswarm plot -> {out_png}")
    print(f"Wrote SHAP metadata -> {metrics_path}")


def main() -> None:
    args = parse_args()
    analysis_label = args.analysis_label
    output_label_dir = "grouped_shap_analysis" if analysis_label == "grouped" else "ungrouped_shap_analysis"

    if analysis_label == "grouped":
        train_path = args.grouped_train_data
        test_path = args.grouped_test_data
    else:
        train_path = args.train_data
        test_path = args.test_data

    # If user requests both best models, attempt to locate them and run SHAP for each.
    if getattr(args, "use_best_both", False):
        print(f"Running SHAP analysis for saved best {analysis_label} models: binary and multiclass.")
        print(f"Best-model directory: {args.grouped_model_dir if analysis_label == 'grouped' else args.best_model_dir}")
        print(f"Output root: {args.output_dir / output_label_dir}")

        best_model_dir = args.grouped_model_dir if analysis_label == "grouped" else args.best_model_dir
        binary_path = best_model_dir / "best_model_binary.joblib"
        multi_path = best_model_dir / "best_model_multiclass.joblib"

        if not binary_path.exists():
            raise FileNotFoundError(f"Best binary model not found at {binary_path}")
        if not multi_path.exists():
            raise FileNotFoundError(f"Best multiclass model not found at {multi_path}")

        # Run SHAP for binary
        print("\n=== SHAP for best binary model ===")
        binary_output_dir = args.output_dir / output_label_dir / "binary"
        binary_png = f"{analysis_label}_shap_best_binary.png"
        run_analysis(binary_path, train_path, test_path, "binary", binary_output_dir, args.max_display, out_png_name=binary_png)

        # Run SHAP for multiclass
        print("\n=== SHAP for best multiclass model ===")
        multi_output_dir = args.output_dir / output_label_dir / "multiclass"
        multi_png = f"{analysis_label}_shap_best_multiclass.png"
        run_analysis(multi_path, train_path, test_path, "multiclass", multi_output_dir, args.max_display, out_png_name=multi_png)
        return

    print(f"Running SHAP analysis for model: {args.model_path}")
    print(f"Task: {args.task} | Output: {args.output_dir}")
    run_analysis(args.model_path, train_path, test_path, args.task, args.output_dir, args.max_display)


if __name__ == "__main__":
    main()
