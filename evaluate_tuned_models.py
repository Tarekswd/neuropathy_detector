"""Re-evaluate saved tuned models and refresh metrics JSON with full metric set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib

from training_utils import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    configure_warnings,
    evaluate_pipeline,
    evaluate_patient_level_from_events,
    load_full_dataset,
)

VALID_MODELS = {"lr", "svm", "rf", "xgb", "catboost"}


def main() -> None:
    configure_warnings()
    tuned_dir = PROJECT_ROOT / "models" / "tuned"
    for model_path in sorted(tuned_dir.glob("*_tuned.joblib")):
        model_name = model_path.stem.replace("_binary_tuned", "").replace("_multiclass_tuned", "")
        if model_name not in VALID_MODELS:
            continue

        payload = joblib.load(model_path)
        task = payload["task"]
        pipeline = payload["model"]

        X_full, y_full, groups_full, _ = load_full_dataset((DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH), task)
        event_metrics = evaluate_pipeline(pipeline, X_full, y_full, task)
        patient_metrics, patient_predictions = evaluate_patient_level_from_events(
            pipeline, X_full, y_full, groups_full, task
        )

        metrics_path = tuned_dir / f"{model_name}_{task}_metrics.json"
        existing = {}
        if metrics_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))

        output = {
            **existing,
            "model": model_name,
            "task": task,
            "event_metrics": event_metrics,
            "patient_metrics": patient_metrics,
            "patient_level_predictions": patient_predictions,
        }
        metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Updated {metrics_path.name}")


if __name__ == "__main__":
    main()
