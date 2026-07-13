from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_builders import build_model
from models.tune_models import FAST_SEARCH_SETTINGS, PARAM_DISTRIBUTIONS, _effective_k_features, _requested_tasks, tune_model
from models.training_utils import (
    build_pipeline,
    evaluate_pipeline,
    evaluate_patient_level_from_events,
    load_full_dataset,
    print_feature_statistics,
    print_metrics,
    subject_cross_validation,
    summarize_feature_statistics,
)

DEFAULT_GROUPED_ROOT = PROJECT_ROOT / "grouped_patient_footprints"
DEFAULT_GROUPED_TRAIN = DEFAULT_GROUPED_ROOT / "grouped_patient_features_train.csv"
DEFAULT_GROUPED_TEST = DEFAULT_GROUPED_ROOT / "grouped_patient_features_test.csv"
DEFAULT_GROUPED_MANIFEST = DEFAULT_GROUPED_ROOT / "grouped_patient_split_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune grouped patient-footprint models with full-data subject-aware CV.")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_GROUPED_TRAIN)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_GROUPED_TEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_GROUPED_MANIFEST)
    parser.add_argument("--task", choices=["all", "binary", "multiclass"], default="all")
    parser.add_argument("--model", choices=["all", "lr", "svm", "rf", "xgb", "catboost"], default="all")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--k-features", type=int, default=20, help="Number of top features to use (default: 20).")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "grouped_tuned")
    parser.add_argument("--best-model-dir", type=Path, default=PROJECT_ROOT / "best model" / "grouped")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.best_model_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_run = _requested_tasks(args.task)
    model_names = list(PARAM_DISTRIBUTIONS) if args.model == "all" else [args.model]

    def filter_event_count(X):
        return X.drop(columns=[col for col in X.columns if col == "event_count"], errors="ignore")

    print("Preprocessing: grouped patient footprints from train/test files")
    print("Tuning: subject-aware CV across the full grouped dataset")
    print("Evaluation: out-of-fold event and patient metrics")

    leaderboard: list[dict] = []
    for task in tasks_to_run:
        print(f"\n=== Grouped tuning task: {task} ===")
        X_full, y_full, groups_full, feature_cols = load_full_dataset([args.train_data, args.test_data], task=task)
        X_full = filter_event_count(X_full)
        feature_cols = [c for c in feature_cols if c != "event_count"] if "event_count" in feature_cols else feature_cols

        for name in model_names:
            print("\n" + "=" * 80)
            print(f"Tuning: {name.upper()} ({task})")
            print("=" * 80)

            best_pipeline, best_params, cv_score = tune_model(name, X_full, y_full, groups_full, task, args)

            fast = FAST_SEARCH_SETTINGS.get(name, {})
            cv_folds = int(fast.get("cv_folds", args.cv_folds) or args.cv_folds)
            k_features = _effective_k_features(args.k_features, X_full.shape[1])
            pipeline_factory = lambda: build_pipeline(
                build_model(name, task, args.random_state),
                k_features,
                X_full.shape[1],
            )
            cv_summary = subject_cross_validation(
                pipeline_factory=pipeline_factory,
                X=X_full,
                y=y_full,
                groups=groups_full,
                task=task,
                cv_folds=cv_folds,
                random_state=args.random_state,
            )

            event_metrics = cv_summary.get("event_metrics")
            patient_metrics = cv_summary.get("patient_metrics")
            patient_predictions = cv_summary.get("patient_predictions", [])
            if event_metrics is None:
                event_metrics = evaluate_pipeline(best_pipeline, X_full, y_full, task)
            if patient_metrics is None:
                patient_metrics, patient_predictions = evaluate_patient_level_from_events(
                    best_pipeline,
                    X_full,
                    y_full,
                    groups_full,
                    task,
                )

            feature_stats = summarize_feature_statistics(X_full, feature_cols)
            print(f"\nBest CV balanced accuracy: {cv_score:.4f}")
            print(f"Best params: {json.dumps(best_params, indent=2)}")
            print_feature_statistics(feature_stats, label=f"{name.upper()} grouped features")
            print("\nCross-validated event metrics:")
            print_metrics(name.upper(), task, event_metrics, label="CV Event Metrics")
            print("\nCross-validated patient metrics:")
            print_metrics(name.upper(), task, patient_metrics, label="CV Patient Metrics")

            artifact = {
                "model": best_pipeline,
                "task": task,
                "model_name": name,
                "features": feature_cols,
                "best_params": best_params,
                "validation": "grouped_patient_footprints_full_subject_cv",
                "normalization": "train_fit_zscore",
                "cv_balanced_accuracy": cv_score,
                "cv_metrics_summary": cv_summary,
                "event_metrics": event_metrics,
                "patient_metrics": patient_metrics,
                "patient_level_predictions": patient_predictions,
                "train_metrics": event_metrics,
                "test_metrics": event_metrics,
            }

            output_path = args.output_dir / f"{name}_{task}_grouped_tuned.joblib"
            joblib.dump(artifact, output_path)

            metrics_path = args.output_dir / f"{name}_{task}_grouped_metrics.json"
            metrics_path.write_text(json.dumps(artifact, default=str, indent=2), encoding="utf-8")

            leaderboard.append(
                {
                    "model": name,
                    "task": task,
                    "output_path": str(output_path),
                    "cv_balanced_accuracy": cv_score,
                    "patient_balanced_accuracy": patient_metrics.get("balanced_accuracy"),
                    "patient_f1_score": patient_metrics.get("f1_score"),
                    "patient_auc": patient_metrics.get("auc"),
                }
            )

    leaderboard.sort(key=lambda row: (row["task"], row["cv_balanced_accuracy"], row["patient_balanced_accuracy"]), reverse=True)
    print("\n=== Grouped tuning leaderboard ===")
    for rank, row in enumerate(leaderboard, start=1):
        auc_str = f"{row['patient_auc']:.4f}" if row["patient_auc"] is not None else "N/A"
        print(
            f"{rank}. {row['model']} ({row['task']}): cv_bal_acc={row['cv_balanced_accuracy']:.4f}, "
            f"patient_bal_acc={row['patient_balanced_accuracy']:.4f}, f1={row['patient_f1_score']:.4f}, auc={auc_str}"
        )

    best_paths = {}
    for task_name in ["binary", "multiclass"]:
        candidates = [row for row in leaderboard if row["task"] == task_name]
        if not candidates:
            continue
        best = candidates[0]
        artifact = joblib.load(best["output_path"])
        artifact["selected_by"] = "cv_balanced_accuracy"
        artifact["selected_task"] = best["task"]
        artifact["selected_model"] = best["model"]
        best_path = args.best_model_dir / f"best_model_{task_name}.joblib"
        joblib.dump(artifact, best_path)
        best_paths[task_name] = {
            "model": best["model"],
            "task": best["task"],
            "output_path": best["output_path"],
            "saved_path": str(best_path),
            "cv_balanced_accuracy": best["cv_balanced_accuracy"],
            "patient_balanced_accuracy": best["patient_balanced_accuracy"],
            "patient_f1_score": best["patient_f1_score"],
            "patient_auc": best["patient_auc"],
        }
        print(f"Saved grouped best {task_name} model -> {best_path}")

    best_metrics_path = args.best_model_dir / "best_model_info.json"
    best_metrics_path.write_text(json.dumps(best_paths, indent=2), encoding="utf-8")
    print(f"Saved grouped best-model info -> {best_metrics_path}")

    try:
        from run_shap_analysis import run_analysis

        for task_name, info in best_paths.items():
            model_path = Path(info["saved_path"])
            shap_out = args.output_dir.parent / "shap_outputs" / f"grouped_{task_name}"
            print(f"Running SHAP for grouped best {task_name} model -> {model_path}")
            try:
                run_analysis(model_path, args.train_data, args.test_data, task_name, shap_out, args.cv_folds, out_png_name=f"shap_grouped_{task_name}.png")
            except Exception as exc:
                print(f"Failed to run SHAP for {task_name}: {exc}")
    except Exception as exc:
        print(f"SHAP analysis skipped: could not import run_shap_analysis ({exc})")


if __name__ == "__main__":
    main()
