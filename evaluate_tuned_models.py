"""Re-evaluate saved tuned models and refresh metrics JSON with full metric set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from models.training_utils import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    configure_warnings,
    evaluate_pipeline,
    load_train_test_datasets,
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

        X_train, X_test, y_train, y_test, _, _, _ = load_train_test_datasets(
            DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH, task
        )

        train_metrics = evaluate_pipeline(pipeline, X_train, y_train, task)
        test_metrics = evaluate_pipeline(pipeline, X_test, y_test, task)

        metrics_path = tuned_dir / f"{model_name}_{task}_metrics.json"
        existing = {}
        if metrics_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))

        output = {
            **existing,
            "model": model_name,
            "task": task,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Updated {metrics_path.name}")


if __name__ == "__main__":
    main()
