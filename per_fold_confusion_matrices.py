from __future__ import annotations

import json
import glob
import sys
from pathlib import Path
from typing import Any

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.model_builders import MULTICLASS_LABELS
from training_utils import load_full_dataset, DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH

PROJECT_ROOT = _PROJECT_ROOT
TUNED_DIR = PROJECT_ROOT / "models" / "tuned"
OUT_DIR = PROJECT_ROOT / "models" / "per_fold_confusion_matrices" / "ungrouped"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from models.tune_models import StratifiedGroupKFold, FAST_SEARCH_SETTINGS
except Exception:
    from sklearn.model_selection import StratifiedGroupKFold
    FAST_SEARCH_SETTINGS = {}


def load_metrics_files():
    tuned_files = sorted(glob.glob(str(TUNED_DIR / "*_tuned.joblib")))
    metrics_files = sorted(glob.glob(str(TUNED_DIR / "*_metrics.json")))
    return tuned_files, metrics_files


def _take_rows(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _labels(task: str) -> list[int]:
    return [0, 1] if task == "binary" else list(MULTICLASS_LABELS)


def _build_fresh_pipeline(artifact: dict, task: str, n_features: int):
    """Rebuild a pipeline from scratch using the artifact's best_params."""
    from models.model_builders import build_model
    from training_utils import build_pipeline

    model_name = artifact.get("model_name") or artifact.get("model")
    if not isinstance(model_name, str):
        # artifact["model"] may be the fitted pipeline itself — clone it
        from sklearn.base import clone
        return clone(artifact["model"])

    pipeline = build_pipeline(build_model(model_name, task, 42), None, n_features)
    best_params = artifact.get("best_params") or {}
    if best_params:
        try:
            pipeline.set_params(**best_params)
        except Exception:
            pass
    return pipeline


def aggregate_subject_data(X, y, groups, feature_cols):
    df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_cols)
    df["subject_id"] = pd.Series(groups, index=df.index)
    df["label"] = pd.Series(y.values if hasattr(y, "values") else y, index=df.index)
    grouped = df.groupby("subject_id")
    grouped_X = grouped[feature_cols].mean()

    def _mode_or_first(s):
        m = s.mode()
        return int(m.iloc[0]) if not m.empty else int(s.iloc[0])

    grouped_y = grouped["label"].agg(_mode_or_first).astype(int)
    grouped_subjects = grouped_X.index.to_numpy(dtype=str)
    return grouped_X, grouped_y, grouped_subjects


def compute_grouped_per_fold_confusion(artifact: dict, X, y, groups, feature_cols, cv, task: str):
    """
    Aggregate events → one row per subject, then run 5-fold CV.
    The 5 val sets cover all subjects, so summing gives the full subject-level matrix.
    """
    grouped_X, grouped_y, grouped_subjects = aggregate_subject_data(X, y, groups, feature_cols)
    labels = _labels(task)
    cms = []
    for train_idx, val_idx in cv.split(grouped_X, grouped_y, grouped_subjects):
        X_tr = _take_rows(grouped_X, train_idx)
        y_tr = _take_rows(grouped_y, train_idx)
        X_val = _take_rows(grouped_X, val_idx)
        y_val = _take_rows(grouped_y, val_idx)

        est = _build_fresh_pipeline(artifact, task, grouped_X.shape[1])
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_val)
        if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        cm = pd.DataFrame(
            confusion_matrix(np.asarray(y_val).ravel(), np.asarray(y_pred).ravel(), labels=labels),
            index=labels,
            columns=labels,
        )
        cms.append(cm)
    return cms


def save_confusion_matrix_png(cm, out_path: Path, title: str) -> None:
    if isinstance(cm, pd.DataFrame):
        labels = [str(x) for x in cm.index.tolist()] if not cm.empty else ["empty"]
        matrix = cm.to_numpy(dtype=int) if not cm.empty else np.zeros((1, 1), dtype=int)
    else:
        matrix = np.asarray(cm, dtype=int)
        labels = [str(i) for i in range(matrix.shape[0])] if matrix.size else ["empty"]
        if not matrix.size:
            matrix = np.zeros((1, 1), dtype=int)

    fig, ax = plt.subplots(figsize=(6, 5))
    if matrix.size and matrix.shape[0] > 0:
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


def _sum_cms(cms: list[pd.DataFrame]) -> pd.DataFrame | None:
    summed = None
    for cm in cms:
        summed = cm.copy() if summed is None else summed.add(cm, fill_value=0).astype(int)
    return summed


def _cm_to_dict(cm: Any) -> dict:
    if isinstance(cm, pd.DataFrame):
        return cm.to_dict()
    try:
        return pd.DataFrame(cm).to_dict()
    except Exception:
        return {}


def run():
    tuned_files, _ = load_metrics_files()
    if not tuned_files:
        print("No tuned artifacts found in", TUNED_DIR)
        return

    processed = 0
    for tf in tuned_files:
        print("Processing:", tf)
        artifact = joblib.load(tf)
        model_name = artifact.get("model_name") or artifact.get("model")
        task = artifact.get("task")
        best_params = artifact.get("best_params")

        train_path = artifact.get("train_data_path") or artifact.get("train_path") or DEFAULT_TRAIN_PATH
        test_path = artifact.get("test_data_path") or artifact.get("test_path") or DEFAULT_TEST_PATH

        # Use 100% of data: train + test combined
        X_full, y_full, groups_full, feature_cols = load_full_dataset(
            [Path(train_path), Path(test_path)], task
        )

        fast = FAST_SEARCH_SETTINGS.get(model_name, {}) if isinstance(model_name, str) else {}
        cv_folds = int(fast.get("cv_folds", 5))
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # --- subject-level per-fold confusion matrices ---
        cms_patient = compute_grouped_per_fold_confusion(
            artifact, X_full, y_full, groups_full, feature_cols, cv, task
        )
        summed_patient = _sum_cms(cms_patient)

        stem = Path(tf).stem
        png_dir = OUT_DIR / "png"
        for fold_idx, cm in enumerate(cms_patient, start=1):
            save_confusion_matrix_png(
                cm, png_dir / f"{stem}_fold{fold_idx}_confusion.png",
                title=f"{stem} fold {fold_idx} (Patient-level)"
            )
        if summed_patient is not None:
            save_confusion_matrix_png(
                summed_patient, png_dir / f"{stem}_summed_confusion.png",
                title=f"{stem} Summed {len(cms_patient)} Folds (Patient-level, n={int(summed_patient.values.sum())})"
            )
            print(f"  Summed patient CM (total={int(summed_patient.values.sum())}) -> {png_dir / f'{stem}_summed_confusion.png'}")

        out = {
            "artifact": tf,
            "model": str(model_name),
            "task": task,
            "best_params": best_params,
            "n_total_patients": int(summed_patient.values.sum()) if summed_patient is not None else 0,
            "cv_folds": cv_folds,
            "per_fold_patient_confusion": [_cm_to_dict(cm) for cm in cms_patient],
            "summed_patient_confusion": _cm_to_dict(summed_patient) if summed_patient is not None else {},
        }

        out_path = OUT_DIR / f"{stem}_per_fold_confusion.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  Wrote -> {out_path}")
        processed += 1

    print(f"Processed {processed} tuned artifacts.")


if __name__ == "__main__":
    run()
