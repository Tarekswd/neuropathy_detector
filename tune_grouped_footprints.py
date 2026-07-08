from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.tune_models import (
    FAST_SEARCH_SETTINGS,
    PARAM_DISTRIBUTIONS,
    _compute_cv_metrics,
    _effective_k_features,
    _requested_tasks,
    tune_model,
)
from models.training_utils import (
    evaluate_pipeline,
    load_train_test_datasets,
    print_feature_statistics,
    print_metrics,
    summarize_feature_statistics,
)

DEFAULT_GROUPED_ROOT = PROJECT_ROOT / "grouped_patient_footprints"
DEFAULT_GROUPED_TRAIN = DEFAULT_GROUPED_ROOT / "grouped_patient_features_train.csv"
DEFAULT_GROUPED_TEST = DEFAULT_GROUPED_ROOT / "grouped_patient_features_test.csv"
DEFAULT_GROUPED_MANIFEST = DEFAULT_GROUPED_ROOT / "grouped_patient_split_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune models on grouped patient footprints.")
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


def load_grouped_split_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _print_cv_summary(cv_summary: dict) -> None:
    # Print two numeric outputs per metric: mean+std and mean-std
    print("\nTrain-split CV metrics (mean+std, mean-std):")
    for metric in ["auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1_score"]:
        data = cv_summary.get(metric, {})
        mean = data.get("mean")
        std = data.get("std")
        if mean is None:
            print(f"- {metric}: N/A")
        else:
            plus = mean + (std if std is not None else 0.0)
            minus = mean - (std if std is not None else 0.0)
            print(f"- {metric}: {plus:.4f}, {minus:.4f}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.best_model_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_run = _requested_tasks(args.task)
    model_names = list(PARAM_DISTRIBUTIONS) if args.model == "all" else [args.model]

    # grouped patient footprints include event_count as metadata, not a model feature
    def filter_event_count(X):
        return X.drop(columns=[col for col in X.columns if col == "event_count"], errors="ignore")

    split_info = load_grouped_split_manifest(args.manifest)

    print("Preprocessing: grouped patient z-normalized training/test data")
    print("Tuning: subject-aware CV within grouped train split")
    print("Final evaluation: fixed grouped holdout test split")
    if split_info:
        print(
            f"Split: {len(split_info.get('train_subjects', []))} train subjects / "
            f"{len(split_info.get('test_subjects', []))} test subjects"
        )

    leaderboard: list[dict] = []
    for task in tasks_to_run:
        print(f"\n=== Grouped tuning task: {task} ===")
        X_train, X_test, y_train, y_test, groups_train, _groups_test, feature_cols = load_train_test_datasets(
            args.train_data,
            args.test_data,
            task=task,
        )
        X_train = filter_event_count(X_train)
        X_test = filter_event_count(X_test)
        feature_cols = [c for c in feature_cols if c != "event_count"] if "event_count" in feature_cols else feature_cols

        for name in model_names:
            print("\n" + "=" * 80)
            print(f"Tuning: {name.upper()} ({task})")
            print("=" * 80)

            best_pipeline, best_params, cv_score = tune_model(
                name, X_train, y_train, groups_train, task, args
            )

            fast = FAST_SEARCH_SETTINGS.get(name, {})
            cv_folds = int(fast.get("cv_folds", args.cv_folds) or args.cv_folds)
            cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)
            cv_metrics = _compute_cv_metrics(best_pipeline, X_train, y_train, groups_train, cv)
            cv_summary = {}
            for key, values in cv_metrics.items():
                if key == "confusion_matrix":
                    continue
                numeric = [v for v in values if v is not None]
                mean = float(np.mean(numeric)) if numeric else None
                std = float(np.std(numeric, ddof=0)) if numeric else None
                cv_summary[key] = {
                    "mean": mean,
                    "std": std,
                    "mean_plus_std": mean + std if mean is not None and std is not None else None,
                    "mean_minus_std": mean - std if mean is not None and std is not None else None,
                    "per_fold": values,
                }

            train_metrics = evaluate_pipeline(best_pipeline, X_train, y_train, task)
            test_metrics = evaluate_pipeline(best_pipeline, X_test, y_test, task)
            train_feature_stats = summarize_feature_statistics(X_train, feature_cols)
            test_feature_stats = summarize_feature_statistics(X_test, feature_cols)

            print(f"\nBest train-split CV balanced accuracy: {cv_score:.4f}")
            print(f"Best params: {json.dumps(best_params, indent=2)}")
            print_feature_statistics(train_feature_stats, label=f"{name.upper()} train features")
            print_feature_statistics(test_feature_stats, label=f"{name.upper()} test features")
            _print_cv_summary(cv_summary)
            print_metrics(name.upper(), task, train_metrics, label="Train (80%)")
            print_metrics(name.upper(), task, test_metrics, label="Test (20%)")

            joblib.dump(
                {
                    "model": best_pipeline,
                    "task": task,
                    "model_name": name,
                    "features": feature_cols,
                    "best_params": best_params,
                    "validation": "grouped_patient_footprints",
                    "normalization": "train_fit_zscore",
                    "cv_balanced_accuracy": cv_score,
                    "cv_metrics_summary": cv_summary,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                },
                args.output_dir / f"{name}_{task}_grouped_tuned.joblib",
            )

            metrics_path = args.output_dir / f"{name}_{task}_grouped_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "model": name,
                        "task": task,
                        "validation": "grouped_patient_footprints",
                        "normalization": "train_fit_zscore",
                        "best_params": best_params,
                        "cv_balanced_accuracy": cv_score,
                        "cv_metrics_summary": cv_summary,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            leaderboard.append(
                {
                    "model": name,
                    "task": task,
                    "output_path": str(args.output_dir / f"{name}_{task}_grouped_tuned.joblib"),
                    "cv_balanced_accuracy": cv_score,
                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "test_f1_score": test_metrics["f1_score"],
                    "test_auc": test_metrics["auc"],
                }
            )

    leaderboard.sort(key=lambda row: (row["task"], row["test_balanced_accuracy"], row["cv_balanced_accuracy"]), reverse=True)
    print("\n=== Grouped tuning leaderboard ===")
    for rank, row in enumerate(leaderboard, start=1):
        auc_str = f"{row['test_auc']:.4f}" if row["test_auc"] is not None else "N/A"
        print(
            f"{rank}. {row['model']} ({row['task']}): cv_bal_acc={row['cv_balanced_accuracy']:.4f}, "
            f"holdout_bal_acc={row['test_balanced_accuracy']:.4f}, f1={row['test_f1_score']:.4f}, auc={auc_str}"
        )

    best_paths = {}
    for task_name in ["binary", "multiclass"]:
        candidates = [row for row in leaderboard if row["task"] == task_name]
        if not candidates:
            continue
        best = candidates[0]
        artifact = joblib.load(best["output_path"])
        artifact["selected_by"] = "holdout_balanced_accuracy"
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
            "test_balanced_accuracy": best["test_balanced_accuracy"],
            "test_f1_score": best["test_f1_score"],
            "test_auc": best["test_auc"],
        }
        print(f"Saved grouped best {task_name} model -> {best_path}")

    best_metrics_path = args.best_model_dir / "best_model_info.json"
    best_metrics_path.write_text(json.dumps(best_paths, indent=2), encoding="utf-8")
    print(f"Saved grouped best-model info -> {best_metrics_path}")

    # Attempt to run SHAP analysis for the selected best models (grouped)
    try:
        from run_shap_analysis import run_analysis

        for task_name, info in best_paths.items():
            model_path = Path(info["saved_path"])
            shap_out = args.output_dir.parent / "shap_outputs" / f"grouped_{task_name}"
            print(f"Running SHAP for grouped best {task_name} model -> {model_path}")
            try:
                run_analysis(model_path, args.train_data, args.test_data, task_name, shap_out, args.cv_folds, out_png_name=f"shap_grouped_{task_name}.png")
            except Exception as e:
                print(f"Failed to run SHAP for {task_name}: {e}")
    except Exception as e:
        print(f"SHAP analysis skipped: could not import run_shap_analysis ({e})")


if __name__ == "__main__":
    main()
