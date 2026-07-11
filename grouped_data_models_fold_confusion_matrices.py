from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_builders import MULTICLASS_LABELS, build_model
from training_utils import build_pipeline, load_full_dataset

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "grouped_tuned"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_train.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "per_fold_confusion_matrices" / "grouped"

METADATA_COLUMNS = {"group", "class_id", "subject_id", "event_id", "source_folder", "event_count"}
BINARY_POSITIVE_GROUPS = {"NL", "NS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-fold confusion matrices for grouped-data models.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _take_rows(data: Any, indices: np.ndarray | list[int]):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _labels(task: str) -> list[int]:
    return [0, 1] if task == "binary" else list(MULTICLASS_LABELS)


def _build_fresh_pipeline(model_name: str, best_params: dict, task: str, n_features: int):
    """Rebuild a pipeline from scratch and apply best_params."""
    pipeline = build_pipeline(build_model(model_name, task, 42), None, n_features)
    if best_params:
        try:
            pipeline.set_params(**best_params)
        except Exception:
            pass
    return pipeline


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_fold_confusions(
    model_name: str,
    best_params: dict[str, Any],
    X,
    y,
    groups,
    cv,
    task: str,
) -> list[dict[str, Any]]:
    """
    Run CV rebuilding the pipeline from scratch each fold.
    The 5 val sets are disjoint and cover 100% of the data,
    so summing the 5 confusion matrices equals the full-data matrix.
    """
    results: list[dict[str, Any]] = []
    labels = _labels(task)

    for fold_index, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_tr = _take_rows(X, train_idx)
        y_tr = _take_rows(y, train_idx)
        X_val = _take_rows(X, val_idx)
        y_val = _take_rows(y, val_idx)

        estimator = _build_fresh_pipeline(model_name, best_params, task, X.shape[1])
        estimator.fit(X_tr, y_tr)
        y_pred = estimator.predict(X_val)

        cm = confusion_matrix(np.asarray(y_val).ravel(), np.asarray(y_pred).ravel(), labels=labels)
        results.append({
            "fold": fold_index,
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "n_val": int(len(val_idx)),
            "n_train": int(len(train_idx)),
        })

    return results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = sorted(glob.glob(str(args.models_dir / "*_grouped_tuned.joblib")))
    if not artifacts:
        print("No grouped tuned artifacts found in", args.models_dir)
        return

    summary: list[dict[str, Any]] = []
    for artifact_path in artifacts:
        artifact = joblib.load(artifact_path)
        if isinstance(artifact, dict):
            model = artifact.get("model")
            task = artifact.get("task")
            model_name = artifact.get("model_name") or Path(artifact_path).stem
        else:
            model = artifact
            task = "binary"
            model_name = Path(artifact_path).stem

        if model is None:
            print("Skipping artifact with no model:", artifact_path)
            continue
        if task not in {"binary", "multiclass"}:
            print("Skipping artifact with unsupported task:", artifact_path)
            continue
        if not isinstance(model_name, str):
            print("Skipping artifact with non-string model_name:", artifact_path)
            continue

        # Use 100% of data: train + test combined
        X_full, y_full, groups_full, feature_cols = load_full_dataset(
            [args.train_data, args.test_data], task=task
        )

        cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        best_params = artifact.get("best_params", {}) if isinstance(artifact, dict) else {}

        fold_results = compute_fold_confusions(
            model_name, best_params, X_full, y_full, groups_full, cv, task
        )

        # Compute summed matrix (should equal full dataset size)
        labels = _labels(task)
        summed_cm = np.zeros((len(labels), len(labels)), dtype=int)
        for fold in fold_results:
            summed_cm += np.asarray(fold["confusion_matrix"], dtype=int)
        n_total = int(summed_cm.sum())

        artifact_name = Path(artifact_path).stem
        output_path = args.output_dir / f"{artifact_name}_fold_confusions.json"
        output_data = {
            "artifact": artifact_name,
            "model_name": model_name,
            "task": task,
            "cv_folds": args.cv_folds,
            "n_total": n_total,
            "feature_columns": feature_cols,
            "folds": fold_results,
            "summed_confusion_matrix": summed_cm.tolist(),
        }

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(output_data, fh, indent=2)

        # Save per-fold PNGs
        png_dir = args.output_dir / "png"
        str_labels = [str(lb) for lb in labels]
        for fold in fold_results:
            cm = np.asarray(fold["confusion_matrix"], dtype=int)
            png_path = png_dir / f"{artifact_name}_fold{fold['fold']}_confusion.png"
            _plot_confusion_matrix(cm, str_labels, png_path,
                                   title=f"{artifact_name} fold {fold['fold']} (n_val={fold['n_val']})")

        # Save summed PNG
        summed_png = png_dir / f"{artifact_name}_summed_confusion.png"
        _plot_confusion_matrix(summed_cm, str_labels, summed_png,
                               title=f"{artifact_name} Summed {args.cv_folds} Folds (n={n_total})")

        summary.append({
            "artifact": artifact_name,
            "task": task,
            "n_total": n_total,
            "output_path": str(output_path),
            "folds": len(fold_results),
            "png_dir": str(png_dir),
        })

        print(f"Saved fold confusion matrices for {artifact_name} -> {output_path}")
        print(f"  Summed matrix total={n_total} -> {summed_png}")
        for fold in fold_results:
            print(f"  fold {fold['fold']}: n_val={fold['n_val']}, n_train={fold['n_train']}")

    summary_path = args.output_dir / "grouped_fold_confusion_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("Saved summary ->", summary_path)


if __name__ == "__main__":
    main()
