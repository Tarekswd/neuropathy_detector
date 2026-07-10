"""
Evaluate tuned (ungrouped) models on grouped patient-level data.
The tuned pipeline already contains its own fitted scaler, so grouped data
is passed directly to pipeline.predict() using the feature columns stored
in each artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training_utils import load_full_dataset

TUNED_DIR = PROJECT_ROOT / "models" / "tuned"
GROUPED_TRAIN = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_train.csv"
GROUPED_TEST = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_test.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "grouped_models_sudden"

VALID_MODELS = {"lr", "svm", "rf", "xgb", "catboost"}


def _compute_metrics(y_true, y_pred, y_proba, task: str) -> dict:
    avg = "binary" if task == "binary" else "weighted"
    metrics: dict = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "auc": None,
        "sensitivity": None,
        "specificity": None,
    }

    if task == "binary":
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        except Exception:
            pass
        if y_proba is not None:
            try:
                metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            except Exception:
                pass

    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grouped_dfs: dict[str, tuple] = {}
    for task in ("binary", "multiclass"):
        X, y, groups, feature_cols = load_full_dataset([GROUPED_TRAIN, GROUPED_TEST], task=task)
        grouped_dfs[task] = (X, y, groups, feature_cols)

    leaderboard: list[dict] = []
    all_results: list[dict] = []

    for artifact_path in sorted(TUNED_DIR.glob("*_tuned.joblib")):
        stem = artifact_path.stem
        for task in ("binary", "multiclass"):
            suffix = f"_{task}_tuned"
            if stem.endswith(suffix):
                algo = stem[: -len(suffix)]
                break
        else:
            print(f"Skipping unrecognised artifact: {artifact_path.name}")
            continue

        if algo not in VALID_MODELS:
            print(f"Skipping unknown model: {algo}")
            continue

        print(f"\n{'='*70}")
        print(f"  {algo.upper()} — {task.upper()}")
        print(f"{'='*70}")

        artifact = joblib.load(artifact_path)
        pipeline = artifact["model"]
        trained_features: list[str] = artifact.get("features") or artifact.get("feature_columns") or []

        X_grouped, y_grouped, _, _ = grouped_dfs[task]

        if trained_features:
            available = [f for f in trained_features if f in X_grouped.columns]
            if not available:
                print("  No matching features found — skipping.")
                continue
            X_eval = X_grouped[available]
        else:
            X_eval = X_grouped

        y_true = np.asarray(y_grouped)
        y_pred = pipeline.predict(X_eval)

        y_proba = None
        try:
            y_proba = pipeline.predict_proba(X_eval)
        except Exception:
            pass

        test_metrics = _compute_metrics(y_true, y_pred, y_proba, task)
        cm = confusion_matrix(y_true, y_pred)
        cm_labels = sorted(np.unique(y_true).tolist())

        print(f"  Balanced accuracy : {test_metrics['balanced_accuracy']:.4f}")
        print(f"  Accuracy          : {test_metrics['accuracy']:.4f}")
        if test_metrics["auc"] is not None:
            print(f"  AUC               : {test_metrics['auc']:.4f}")
        print(f"  Sensitivity       : {test_metrics['sensitivity']}")
        print(f"  Specificity       : {test_metrics['specificity']}")
        print(f"  F1                : {test_metrics['f1_score']:.4f}")
        print(f"  Confusion matrix:\n{cm}")

        result = {
            "model": algo,
            "task": task,
            "model_type": "ungrouped_tuned_tested_on_grouped",
            "description": "Tuned (ungrouped event-level) model evaluated on grouped patient-level data",
            "artifact": str(artifact_path),
            "features_used": list(X_eval.columns),
            "test_metrics": test_metrics,
            "confusion_matrix": {"labels": cm_labels, "matrix": cm.tolist()},
        }

        metrics_path = OUTPUT_DIR / f"{algo}_{task}_grouped_metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  Saved -> {metrics_path.name}")

        all_results.append(result)
        leaderboard.append({
            "model": algo,
            "task": task,
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "accuracy": test_metrics["accuracy"],
            "f1_score": test_metrics["f1_score"],
            "auc": test_metrics["auc"],
        })

    leaderboard.sort(key=lambda r: (r["task"], r["balanced_accuracy"]), reverse=True)
    (OUTPUT_DIR / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "all_grouped_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"\nSaved leaderboard -> {OUTPUT_DIR / 'leaderboard.json'}")
    print(f"Saved all results -> {OUTPUT_DIR / 'all_grouped_results.json'}")
    print(f"\nDone. Evaluated {len(all_results)} model/task combinations.")


if __name__ == "__main__":
    main()
