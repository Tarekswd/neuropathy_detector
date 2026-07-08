"""Generate confusion matrix PNGs for grouped-tuned model artifacts.

Usage:
  python generate_grouped_confusion_matrices.py --models-dir models/grouped_tuned --output-dir output_plots/grouped_confusion_matrices

The script:
- finds all `*_grouped_metrics.json` and `*_grouped.joblib` artifacts
- loads the corresponding model artifact (`joblib`), the train/test grouped CSVs, and computes confusion matrices (test and train)
- renders and saves PNGs per model (train_confusion_MATRIX.png and test_confusion_MATRIX.png)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "grouped_tuned"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_plots" / "grouped_confusion_matrices"
DEFAULT_TRAIN = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_train.csv"
DEFAULT_TEST = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_test.csv"

METADATA_COLUMNS = {"group", "class_id", "subject_id", "event_id", "source_folder", "event_count"}
BINARY_POSITIVE_GROUPS = {"NL", "NS"}
BINARY_LABELS = [0, 1]
BINARY_LABEL_NAMES = ["non_neuropathy", "neuropathy"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    return parser.parse_args()


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns if c not in METADATA_COLUMNS]


def _plot_and_save(cm, labels, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    models_dir = args.models_dir
    out_dir = args.output_dir

    train_df = _load_frame(args.train)
    test_df = _load_frame(args.test)

    feature_cols = _feature_cols(train_df)
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # find joblib artifacts in models_dir
    artifacts = list(models_dir.glob("*_grouped*.joblib"))
    if not artifacts:
        print("No grouped joblib artifacts found in", models_dir)
        return

    for artifact_path in artifacts:
        try:
            data = joblib.load(artifact_path)
        except Exception as e:
            print("Failed to load", artifact_path, e)
            continue

        model = data.get("model") if isinstance(data, dict) else data
        model_name = artifact_path.stem.replace("_grouped_tuned", "").replace("_grouped", "")
        task_is_binary = "binary" in artifact_path.stem

        if task_is_binary:
            labels = [0, 1]
            label_names = ["non_neuropathy", "neuropathy"]
        else:
            labels = sorted(train_df["class_id"].unique().tolist())
            label_names = [str(label) for label in labels]

        # predict and plot for train
        try:
            if task_is_binary:
                y_train = train_df["group"].isin(BINARY_POSITIVE_GROUPS).astype(int)
            else:
                y_train = train_df["class_id"].astype(int)
            y_pred_train = model.predict(X_train)
            cm_train = np.array([[0]])
            try:
                from sklearn.metrics import confusion_matrix

                cm_train = confusion_matrix(y_train, y_pred_train, labels=labels)
            except Exception:
                pass
            out_train = out_dir / f"{model_name}_train_confusion.png"
            _plot_and_save(cm_train, label_names, out_train, f"{model_name} - Train Confusion Matrix")
        except Exception as e:
            print("Train plot failed for", artifact_path, e)

        # predict and plot for test
        try:
            if task_is_binary:
                y_test = test_df["group"].isin(BINARY_POSITIVE_GROUPS).astype(int)
            else:
                y_test = test_df["class_id"].astype(int)
            y_pred_test = model.predict(X_test)
            cm_test = np.array([[0]])
            try:
                from sklearn.metrics import confusion_matrix

                cm_test = confusion_matrix(y_test, y_pred_test, labels=labels)
            except Exception:
                pass
            out_test = out_dir / f"{model_name}_test_confusion.png"
            _plot_and_save(cm_test, label_names, out_test, f"{model_name} - Test Confusion Matrix")
        except Exception as e:
            print("Test plot failed for", artifact_path, e)

    print("Done. Saved confusion matrices to", out_dir)


if __name__ == "__main__":
    main()
