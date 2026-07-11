from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_builders import build_model
from models.training_utils import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    build_pipeline,
    configure_warnings,
    evaluate_pipeline,
    evaluate_patient_level_from_events,
    load_full_dataset,
    load_split_manifest,
    load_train_test_datasets,
    print_feature_statistics,
    print_metrics,
    summarize_feature_statistics,
    _effective_k_features,
)

# Compact search spaces for small tabular dataset (20 features, ~430 train events).
PARAM_DISTRIBUTIONS = {
    "lr": {
        "binary": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "multiclass": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__solver": ["lbfgs"],
        },
    },
    "svm": {
        "model__C": [0.1, 1.0, 10.0, 50.0],
        "model__gamma": ["scale", 0.01, 0.1],
        "model__kernel": ["rbf", "linear"],
    },
    "rf": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [8, 12, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
    "xgb": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0],
    },
    "catboost": {
        "model__iterations": [150, 250, 350],
        "model__depth": [4, 6],
        "model__learning_rate": [0.05, 0.1],
        "model__l2_leaf_reg": [3.0, 5.0],
    },
}

# CatBoost already uses all cores; avoid nested parallel CV jobs.
SERIAL_SEARCH_MODELS = {"catboost"}
FAST_SEARCH_SETTINGS = {
    "catboost": {"cv_folds": 3, "n_iter_cap": 9},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with subject-aware cross-validation on all available data.")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--task", choices=["all", "binary", "multiclass"], default="all")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--k-features", type=int, default=None, help="Top K features (default: all).")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "lr", "svm", "rf", "xgb", "catboost"])
    parser.add_argument("--n-iter", type=int, default=20, help="Unused; grid search evaluates the full parameter grid.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "tuned")
    parser.add_argument("--best-model-dir", type=Path, default=PROJECT_ROOT / "best model")
    return parser.parse_args()


def _requested_tasks(task_arg: str) -> list[str]:
    if task_arg == "all":
        return ["binary", "multiclass"]
    return [task_arg]


def get_param_distribution(name: str, task: str) -> dict:
    params = PARAM_DISTRIBUTIONS[name]
    if name == "lr":
        return dict(params[task])
    return dict(params)


def tune_model(
    name: str,
    X_train,
    y_train,
    groups_train,
    task: str,
    args,
):
    n_features = X_train.shape[1]
    k_features = _effective_k_features(args.k_features, n_features)
    base_model = build_model(name, task, args.random_state)
    pipeline = build_pipeline(base_model, k_features, n_features)

    param_dist = get_param_distribution(name, task)
    if k_features < n_features:
        param_dist["feature_selection__k"] = sorted(
            {max(5, k_features // 2), k_features, min(n_features, k_features + 5)}
        )

    fast = FAST_SEARCH_SETTINGS.get(name, {})
    cv_folds = int(fast.get("cv_folds", args.cv_folds) or args.cv_folds)

    cv = StratifiedGroupKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )
    search_jobs = 1 if name in SERIAL_SEARCH_MODELS else -1
    search = GridSearchCV(
        pipeline,
        param_grid=param_dist,
        scoring=make_scorer(balanced_accuracy_score),
        cv=cv,
        n_jobs=search_jobs,
        refit=True,
        verbose=0,
        error_score=0.0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        search.fit(X_train, y_train, groups=groups_train)

    return search.best_estimator_, search.best_params_, float(search.best_score_)


def _take_rows(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _aggregate_to_patient_level(X, y, groups):
    """Collapse event rows → one row per subject (mean features, majority label)."""
    import pandas as pd
    df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["__subject__"] = np.asarray(groups)
    df["__label__"] = np.asarray(y.values if hasattr(y, "values") else y)
    grp = df.groupby("__subject__")
    feat_cols = [c for c in df.columns if c not in ("__subject__", "__label__")]
    X_pat = grp[feat_cols].mean()

    def _mode(s):
        m = s.mode()
        return int(m.iloc[0]) if not m.empty else int(s.iloc[0])

    y_pat = grp["__label__"].agg(_mode).astype(int)
    groups_pat = X_pat.index.to_numpy(dtype=str)
    return X_pat, y_pat, groups_pat


def _compute_cv_metrics(estimator, X, y, groups, cv):
    """Run manual CV to collect per-fold metrics (returns dict of lists)."""
    metrics = {
        "auc": [],
        "balanced_accuracy": [],
        "accuracy": [],
        "precision": [],
        "sensitivity": [],
        "specificity": [],
        "f1_score": [],
        "confusion_matrix": [],
    }

    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr, X_val = _take_rows(X, train_idx), _take_rows(X, val_idx)
        y_tr, y_val = _take_rows(y, train_idx), _take_rows(y, val_idx)

        est = clone(estimator)
        # fit on the fold's training portion
        est.fit(X_tr, y_tr)

        # predictions
        y_pred = est.predict(X_val)

        # probabilities for AUC if available
        y_prob = None
        try:
            y_prob = est.predict_proba(X_val)
        except (AttributeError, ValueError, TypeError):
            y_prob = None

        # AUC (handle binary / multiclass)
        try:
            if y_prob is not None:
                if len(y_prob.shape) == 1 or y_prob.shape[1] == 2:
                    # binary
                    auc = roc_auc_score(y_val, y_prob[:, 1])
                else:
                    auc = roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted")
            else:
                auc = None
        except (AttributeError, ValueError, TypeError):
            auc = None

        bal_acc = balanced_accuracy_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)

        # precision/recall/f1: use weighted average for multiclass, binary default
        avg = "binary" if len(set(y)) == 2 else "weighted"
        precision = precision_score(y_val, y_pred, average=avg, zero_division=0)
        recall = recall_score(y_val, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_val, y_pred, average=avg, zero_division=0)

        # specificity: compute from confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        if cm.size == 4:
            # binary
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # multiclass: compute per-class specificity and weight by support
            supports = cm.sum(axis=1)
            total = cm.sum()
            spec_per_class = []
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = total - (tp + fp + fn)
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                spec_per_class.append((spec, supports[i]))
            # weighted by support
            if total > 0:
                specificity = sum(s * sup for s, sup in spec_per_class) / total
            else:
                specificity = 0.0

        metrics["auc"].append(auc)
        metrics["balanced_accuracy"].append(bal_acc)
        metrics["accuracy"].append(acc)
        metrics["precision"].append(precision)
        metrics["sensitivity"].append(recall)
        metrics["specificity"].append(specificity)
        metrics["f1_score"].append(f1)
        metrics["confusion_matrix"].append(cm.tolist())

    return metrics


def _summarize_cv_metrics(cv_metrics: dict) -> dict:
    """Convert raw per-fold lists into {mean, std, per_fold, ...} summary dict."""
    summary = {}
    for key, values in cv_metrics.items():
        if key == "confusion_matrix":
            mats = [np.array(m) for m in values]
            max_dim = max(m.shape[0] for m in mats)
            stacked = np.zeros((len(mats), max_dim, max_dim), dtype=float)
            for i, m in enumerate(mats):
                stacked[i, : m.shape[0], : m.shape[1]] = m
            summary[key] = {
                "per_fold": [m.tolist() for m in mats],
                "mean_matrix": stacked.mean(axis=0).tolist(),
                "std_matrix": stacked.std(axis=0).tolist(),
                "sum_matrix": stacked.sum(axis=0).tolist(),
            }
        else:
            numeric = [v for v in values if v is not None]
            mean = float(np.mean(numeric)) if numeric else None
            std = float(np.std(numeric, ddof=0)) if numeric else None
            summary[key] = {
                "mean": mean,
                "std": std,
                "mean_plus_std": mean + std if mean is not None and std is not None else None,
                "mean_minus_std": mean - std if mean is not None and std is not None else None,
                "per_fold": values,
            }
    return summary


def _print_cv_summary(cv_summary: dict) -> None:
    for metric in ["auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1_score"]:
        m = cv_summary.get(metric, {})
        mean = m.get("mean") if m else None
        std = m.get("std", 0.0) if m else None
        if mean is not None:
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
        else:
            print(f"  {metric}: N/A")
    cm_info = cv_summary.get("confusion_matrix", {})
    if cm_info:
        print("  Confusion matrix (mean):")
        for row in cm_info.get("mean_matrix", []):
            print(f"    {[round(v, 1) for v in row]}")
        print("  Confusion matrix (sum across folds):")
        for row in cm_info.get("sum_matrix", []):
            print(f"    {[int(v) for v in row]}")


def main():
    configure_warnings()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.best_model_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_run = _requested_tasks(args.task)
    model_names = list(PARAM_DISTRIBUTIONS) if args.model == "all" else [args.model]
    leaderboard = []

    for task in tasks_to_run:
        _, X_test, _, _, _, _, feature_cols = load_train_test_datasets(
            args.train_data,
            args.test_data,
            task,
        )
        data_paths = [args.train_data, args.test_data]
        X_full, y_full, groups_full, full_feature_cols = load_full_dataset(data_paths, task)
        split_info = load_split_manifest()

        print("Preprocessing: z-normalization fit on all available data")
        print("Tuning: subject-level CV on all available data")
        print("Final evaluation: patient-level metrics from out-of-fold predictions")
        if split_info:
            print(
                f"Split: {len(split_info.get('train_subjects', []))} train subjects / "
                f"{len(split_info.get('test_subjects', []))} test subjects"
            )

        for name in model_names:
            print("\n" + "=" * 80)
            print(f"Tuning: {name.upper()} ({task})")
            print("=" * 80)

            best_pipeline, best_params, cv_score = tune_model(
                name, X_full, y_full, groups_full, task, args
            )

            # recompute CV folds used (honor FAST_SEARCH_SETTINGS)
            fast = FAST_SEARCH_SETTINGS.get(name, {})
            cv_folds = int(fast.get("cv_folds", args.cv_folds) or args.cv_folds)
            cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)

            # ------------------------------------------------------------------ #
            # Phase 1: event-level CV metrics
            # ------------------------------------------------------------------ #
            print("\nComputing per-fold event-level CV metrics...")
            cv_metrics = _compute_cv_metrics(best_pipeline, X_full, y_full, groups_full, cv)
            cv_summary = _summarize_cv_metrics(cv_metrics)

            print("\nEvent-level CV (mean ± std):")
            _print_cv_summary(cv_summary)

            # ------------------------------------------------------------------ #
            # Phase 2: patient-level CV — retrain on aggregated subject rows
            # ------------------------------------------------------------------ #
            print("\nComputing per-fold patient-level CV metrics...")
            X_pat, y_pat, groups_pat = _aggregate_to_patient_level(X_full, y_full, groups_full)
            patient_cv_metrics = _compute_cv_metrics(best_pipeline, X_pat, y_pat, groups_pat, cv)
            patient_cv_summary = _summarize_cv_metrics(patient_cv_metrics)

            print("\nPatient-level CV (mean ± std):")
            _print_cv_summary(patient_cv_summary)

            event_metrics = evaluate_pipeline(best_pipeline, X_full, y_full, task)
            patient_metrics, patient_predictions = evaluate_patient_level_from_events(
                best_pipeline, X_full, y_full, groups_full, task
            )

            print("\n" + "-" * 60)
            print("Patient-level evaluation from event-level predictions")
            print("-" * 60)
            print_metrics(name.upper(), task, patient_metrics, label="Patient-level (OOF aggregation)")

            train_feature_stats = summarize_feature_statistics(X_full, full_feature_cols)
            test_feature_stats = summarize_feature_statistics(X_test, feature_cols)
            print(f"\nBest subject CV balanced accuracy: {cv_score:.4f}")
            print(f"Best params: {json.dumps(best_params, indent=2)}")
            print_feature_statistics(train_feature_stats, label=f"{name.upper()} all-data features")
            print_feature_statistics(test_feature_stats, label=f"{name.upper()} test features")
            print_metrics(name.upper(), task, event_metrics, label="Event-level (full data)")
            output_path = args.output_dir / f"{name}_{task}_tuned.joblib"
            joblib.dump(
                {
                    "model": best_pipeline,
                    "task": task,
                    "model_name": name,
                    "features": full_feature_cols,
                    "best_params": best_params,
                    "validation": "subject_aware_cv_all_data",
                    "normalization": "all_data_fit_zscore",
                    "cv_balanced_accuracy": cv_score,
                    "cv_metrics_summary": cv_summary,
                    "patient_cv_metrics_summary": patient_cv_summary,
                    "event_metrics": event_metrics,
                    "patient_metrics": patient_metrics,
                    "patient_level_predictions": patient_predictions,
                    "feature_statistics": {
                        "all_data": train_feature_stats,
                        "test": test_feature_stats,
                    },
                },
                output_path,
            )
            print(f"\nSaved tuned model -> {output_path}")

            metrics_path = args.output_dir / f"{name}_{task}_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "model": name,
                        "task": task,
                        "validation": "subject_aware_cv_all_data",
                        "normalization": "all_data_fit_zscore",
                        "best_params": best_params,
                        "cv_balanced_accuracy": cv_score,
                        "cv_metrics_summary": cv_summary,
                        "patient_cv_metrics_summary": patient_cv_summary,
                        "event_metrics": event_metrics,
                        "patient_metrics": patient_metrics,
                        "patient_level_predictions": patient_predictions,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            try:
                grouped_output_dir = PROJECT_ROOT / "models" / "grouped_models_sudden"
                grouped_output_dir.mkdir(parents=True, exist_ok=True)

                grouped_artifact_path = grouped_output_dir / f"{name}_{task}_grouped_model.joblib"
                joblib.dump(
                    {
                        "model": best_pipeline,
                        "task": task,
                        "model_name": name,
                        "features": full_feature_cols,
                        "best_params": best_params,
                        "note": "Ungrouped model trained with subject-aware CV over all event rows; patient metrics aggregate out-of-fold predictions by patient.",
                        "patient_metrics": patient_metrics,
                        "patient_level_predictions": patient_predictions,
                    },
                    grouped_artifact_path,
                )

                grouped_metrics_path = grouped_output_dir / f"{name}_{task}_grouped_metrics.json"
                grouped_metrics_path.write_text(
                    json.dumps(
                        {
                            "model": name,
                            "task": task,
                            "note": "Ungrouped model trained with subject-aware CV over all event rows; patient metrics aggregate out-of-fold predictions by patient.",
                            "patient_metrics": patient_metrics,
                            "patient_level_predictions": patient_predictions,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print(f"Saved grouped model artifact -> {grouped_artifact_path}")
                print(f"Saved grouped metrics -> {grouped_metrics_path}")
            except (OSError, TypeError, ValueError):
                print("Warning: failed to save grouped model artifacts.")

            leaderboard.append(
                {
                    "model": name,
                    "task": task,
                    "output_path": str(output_path),
                    "cv_balanced_accuracy": cv_score,
                    "event_balanced_accuracy": event_metrics["balanced_accuracy"],
                    "patient_balanced_accuracy": patient_metrics["balanced_accuracy"],
                    "patient_f1_score": patient_metrics["f1_score"],
                    "patient_auc": patient_metrics["auc"],
                }
            )

    if not leaderboard:
        raise RuntimeError("No models were trained.")

    leaderboard.sort(key=lambda row: (row["patient_balanced_accuracy"], row["cv_balanced_accuracy"]), reverse=True)
    print("\n" + "=" * 80)
    print("LEADERBOARD (by patient-level CV balanced accuracy)")
    print("=" * 80)
    for rank, row in enumerate(leaderboard, start=1):
        auc = row["patient_auc"]
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(
            f"{rank}. {row['model']} ({row['task']}): "
            f"cv_bal_acc={row['cv_balanced_accuracy']:.4f}, "
            f"patient_bal_acc={row['patient_balanced_accuracy']:.4f}, "
            f"f1={row['patient_f1_score']:.4f}, "
            f"auc={auc_str}"
        )

    # select best model per task (binary and multiclass) by holdout balanced accuracy
    best_models_info = {}
    for task_name in ["binary", "multiclass"]:
        candidates = [row for row in leaderboard if row["task"] == task_name]
        if not candidates:
            continue
        # pick the top candidate for this task
        candidates.sort(key=lambda r: (r["patient_balanced_accuracy"], r["cv_balanced_accuracy"]), reverse=True)
        best = candidates[0]
        artifact = joblib.load(best["output_path"])
        artifact["selected_by"] = "patient_cv_balanced_accuracy"
        artifact["selected_task"] = best["task"]
        artifact["selected_model"] = best["model"]

        model_filename = f"best_model_{task_name}.joblib"
        model_path = args.best_model_dir / model_filename
        joblib.dump(artifact, model_path)

        best_models_info[task_name] = {
            "model": best["model"],
            "task": best["task"],
            "output_path": best["output_path"],
            "saved_path": str(model_path),
            "cv_balanced_accuracy": best["cv_balanced_accuracy"],
            "patient_balanced_accuracy": best["patient_balanced_accuracy"],
            "patient_f1_score": best["patient_f1_score"],
            "patient_auc": best["patient_auc"],
        }

        print(f"Saved best {task_name} model -> {model_path}")

    # write combined best-model info
    best_metrics_path = args.best_model_dir / "best_model_info.json"
    best_metrics_path.write_text(json.dumps(best_models_info, indent=2), encoding="utf-8")
    print(f"\nSaved best-model info -> {best_metrics_path}")


if __name__ == "__main__":
    main()
