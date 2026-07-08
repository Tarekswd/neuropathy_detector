from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import pandas as pd

from models.model_builders import MULTICLASS_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNED_DIR = PROJECT_ROOT / "models" / "tuned"
OUT_DIR = PROJECT_ROOT / "models" / "per_fold_confusion_matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# We will reuse tune_models' CV splitter to ensure same folds when possible
try:
    from models.tune_models import StratifiedGroupKFold, FAST_SEARCH_SETTINGS
except Exception:
    from sklearn.model_selection import StratifiedGroupKFold
    FAST_SEARCH_SETTINGS = {}


def load_metrics_files():
    # target tuned outputs (joblib) or metrics jsons
    tuned_files = sorted(glob.glob(str(TUNED_DIR / "*_tuned.joblib")))
    metrics_files = sorted(glob.glob(str(TUNED_DIR / "*_metrics.json")))
    return tuned_files, metrics_files


def _take_rows(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _full_confusion_frame(y_true, y_pred, task: str) -> tuple[pd.DataFrame, list[int]]:
    labels = [0, 1] if task == "binary" else list(MULTICLASS_LABELS)
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    cm = pd.DataFrame(
        confusion_matrix(y_true_arr, y_pred_arr, labels=labels),
        index=labels,
        columns=labels,
    )
    return cm, labels


def compute_per_fold_confusion(estimator, X, y, groups, cv, task: str):
    cms = []
    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr, X_val = _take_rows(X, train_idx), _take_rows(X, val_idx)
        y_tr, y_val = _take_rows(y, train_idx), _take_rows(y, val_idx)
        est = clone(estimator)
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_val)
        if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
            y_pred = y_pred.ravel()
        y_true_series = pd.Series(y_val).ravel() if not isinstance(y_val, pd.Series) else y_val
        cm, _ = _full_confusion_frame(y_true_series, y_pred, task)
        cms.append(cm)
    return cms


def aggregate_subject_data(X, y, groups, feature_cols):
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(X, columns=feature_cols)
    df["subject_id"] = pd.Series(groups)
    df["label"] = pd.Series(y.values if hasattr(y, "values") else y)
    grouped = df.groupby("subject_id")
    grouped_X = grouped[feature_cols].mean()
    def _mode_or_first(s):
        m = s.mode()
        return int(m.iloc[0]) if not m.empty else int(s.iloc[0])
    grouped_y = grouped["label"].agg(_mode_or_first).astype(int)
    grouped_subjects = grouped_X.index.to_numpy(dtype=str)
    return grouped_X, grouped_y, grouped_subjects


def compute_grouped_per_fold_confusion(estimator, X, y, groups, feature_cols, cv, task: str):
    grouped_X, grouped_y, grouped_subjects = aggregate_subject_data(X, y, groups, feature_cols)
    cms = []
    for train_idx, val_idx in cv.split(grouped_X, grouped_y, grouped_subjects):
        X_tr = _take_rows(grouped_X, train_idx)
        y_tr = _take_rows(grouped_y, train_idx)
        X_val = _take_rows(grouped_X, val_idx)
        y_val = _take_rows(grouped_y, val_idx)
        est = clone(estimator)
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_val)
        if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
            y_pred = y_pred.ravel()
        y_true_series = pd.Series(y_val).ravel() if not isinstance(y_val, pd.Series) else y_val
        cm, _ = _full_confusion_frame(y_true_series, y_pred, task)
        cms.append(cm)
    return cms


def save_confusion_matrix_png(cm, out_path: Path, title: str) -> None:
    if isinstance(cm, pd.DataFrame):
        if cm.empty:
            labels = ["empty"]
            matrix = np.zeros((1, 1), dtype=int)
        else:
            labels = [str(x) for x in cm.index.tolist()]
            matrix = cm.to_numpy(dtype=int)
    else:
        matrix = np.asarray(cm, dtype=int)
        if matrix.size == 0:
            labels = ["empty"]
            matrix = np.zeros((1, 1), dtype=int)
        else:
            labels = [str(i) for i in range(matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    if matrix.size and matrix.shape[0] > 0 and matrix.shape[1] > 0:
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
    else:
        ax.text(0.5, 0.5, "No validation samples", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run():
    tuned_files, metrics_files = load_metrics_files()
    if not tuned_files and not metrics_files:
        print("No tuned artifacts found in", TUNED_DIR)
        return

    # prefer joblib tuned artifacts; fall back to metric entries to locate models in other folders
    processed = 0
    for tf in tuned_files:
        print("Processing:", tf)
        artifact = joblib.load(tf)
        model = artifact.get("model_name") or artifact.get("model")
        task = artifact.get("task")
        features = artifact.get("features")
        best_params = artifact.get("best_params")

        # try to locate original dataset used during tuning
        from models.training_utils import load_train_test_datasets, DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH

        # many artifacts store the train/test paths used during tuning; try to read them
        train_path = artifact.get("train_data_path") or artifact.get("train_path")
        test_path = artifact.get("test_data_path") or artifact.get("test_path")
        if train_path is None:
            train_path = DEFAULT_TRAIN_PATH
        if test_path is None:
            test_path = DEFAULT_TEST_PATH

        X_train, X_test, y_train, y_test, groups_train, groups_test, feature_cols = load_train_test_datasets(
            Path(train_path), Path(test_path), task
        )

        # determine cv folds used
        fast = FAST_SEARCH_SETTINGS.get(artifact.get("model_name") or artifact.get("model_name"), {}) if FAST_SEARCH_SETTINGS else {}
        cv_folds = int(fast.get("cv_folds", 5))
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # compute per-fold confusion matrices at event-level
        cms_event = compute_per_fold_confusion(artifact["model"], X_train, y_train, groups_train, cv, task)

        # compute per-fold grouped confusion using subject-level aggregation before CV
        cms_grouped = compute_grouped_per_fold_confusion(
            artifact["model"],
            X_train,
            y_train,
            groups_train,
            feature_cols,
            cv,
            task,
        )

        def _cm_to_dict(cm):
            if isinstance(cm, pd.DataFrame):
                return cm.to_dict()
            if isinstance(cm, dict):
                return cm
            try:
                return pd.DataFrame(cm).to_dict()
            except Exception:
                return {}

        png_dir = OUT_DIR / "png" / "event"
        for fold_idx, cm in enumerate(cms_event, start=1):
            png_path = png_dir / f"{Path(tf).stem}_fold{fold_idx}_confusion.png"
            save_confusion_matrix_png(cm, png_path, title=f"{Path(tf).stem} fold {fold_idx}")

        out = {
            "artifact": tf,
            "model": model,
            "task": task,
            "best_params": best_params,
            "per_fold_event_confusion": [_cm_to_dict(cm) for cm in cms_event],
            "per_fold_grouped_confusion": [_cm_to_dict(cm) for cm in cms_grouped],
        }

        out_path = OUT_DIR / (Path(tf).stem + "_per_fold_confusion.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("Wrote ->", out_path)
        processed += 1

    print(f"Processed {processed} tuned artifacts.")


if __name__ == "__main__":
    run()
