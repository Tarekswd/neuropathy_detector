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
from models.training_utils import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "grouped_tuned"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "grouped_models_sudden" / "grouped_fold_confusion_matrices"

METADATA_COLUMNS = {"group", "class_id", "subject_id", "event_id", "source_folder", "event_count"}
BINARY_POSITIVE_GROUPS = {"NL", "NS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-fold confusion matrices for grouped-data models.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _take_rows(data: Any, indices: np.ndarray | list[int]):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _load_grouped_train(train_path: Path, task: str):
    df = pd.read_csv(train_path)
    if task == "binary":
        y = df["group"].isin(BINARY_POSITIVE_GROUPS).astype(int)
    elif task == "multiclass":
        y = df["class_id"].astype(int)
    else:
        raise ValueError(f"Unknown task: {task}")

    feature_columns = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in METADATA_COLUMNS
    ]
    X = df[feature_columns]
    groups = df["subject_id"].astype(str).values
    return X, y, groups, feature_columns


def _labels_from_values(y: Any) -> list[int]:
    arr = np.asarray(y)
    if arr.ndim > 1:
        arr = arr.ravel()
    return [int(x) for x in arr]


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_fold_confusions(model_name: str, best_params: dict[str, Any], X, y, groups, cv, task: str):
    results: list[dict[str, Any]] = []
    labels = [0, 1] if task == "binary" else list(MULTICLASS_LABELS)

    for fold_index, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_tr = _take_rows(X, train_idx)
        y_tr = _take_rows(y, train_idx)
        X_val = _take_rows(X, val_idx)
        y_val = _take_rows(y, val_idx)

        estimator = build_pipeline(build_model(model_name, task, 42), None, X.shape[1])
        if best_params:
            estimator.set_params(**best_params)
        estimator.fit(X_tr, y_tr)
        y_pred = estimator.predict(X_val)

        cm = confusion_matrix(np.asarray(y_val).ravel(), np.asarray(y_pred).ravel(), labels=labels)

        results.append(
            {
                "fold": fold_index,
                "labels": labels,
                "confusion_matrix": cm.tolist(),
                "n_val": int(len(val_idx)),
                "n_train": int(len(train_idx)),
            }
        )

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
            model_name = artifact.get("model_name") or artifact_path.stem
        else:
            model = artifact
            task = "binary"
            model_name = artifact_path.stem

        if model is None:
            print("Skipping artifact with no model:", artifact_path)
            continue
        if task not in {"binary", "multiclass"}:
            print("Skipping artifact with unsupported task:", artifact_path)
            continue

        X_train, y_train, groups_train, feature_cols = _load_grouped_train(args.train_data, task)
        cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

        best_params = artifact.get("best_params", {}) if isinstance(artifact, dict) else {}
        fold_results = compute_fold_confusions(model_name, best_params, X_train, y_train, groups_train, cv, task)
        artifact_name = Path(artifact_path).stem
        output_path = args.output_dir / f"{artifact_name}_fold_confusions.json"
        output_data = {
            "artifact": artifact_name,
            "model_name": model_name,
            "task": task,
            "cv_folds": args.cv_folds,
            "feature_columns": feature_cols,
            "folds": fold_results,
        }

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(output_data, fh, indent=2)

        png_dir = args.output_dir / "png"
        for fold in fold_results:
            labels = [str(label) for label in fold["labels"]]
            cm = np.asarray(fold["confusion_matrix"], dtype=int)
            png_path = png_dir / f"{artifact_name}_fold{fold['fold']}_confusion.png"
            _plot_confusion_matrix(
                cm,
                labels,
                png_path,
                title=f"{artifact_name} fold {fold['fold']}"
            )

        summary.append({
            "artifact": artifact_name,
            "task": task,
            "output_path": str(output_path),
            "folds": len(fold_results),
            "png_dir": str(png_dir),
        })

        print(f"Saved per-fold grouped confusion matrices for {artifact_name} -> {output_path}")
        print(f"Saved PNGs for {artifact_name} -> {png_dir}")
        for fold in fold_results:
            print(f"{artifact_name} fold {fold['fold']} labels={fold['labels']} n_val={fold['n_val']} n_train={fold['n_train']}")
            print(json.dumps(fold["confusion_matrix"], indent=2))

    summary_path = args.output_dir / "grouped_fold_confusion_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("Saved summary ->", summary_path)


if __name__ == "__main__":
    main()
