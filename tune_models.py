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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_builders import build_model
from models.training_utils import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    build_pipeline,
    configure_warnings,
    evaluate_pipeline,
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
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with fixed 80/20 subject split.")
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
        except Exception:
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
        except Exception:
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


def main():
    configure_warnings()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.best_model_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_run = _requested_tasks(args.task)
    model_names = list(PARAM_DISTRIBUTIONS) if args.model == "all" else [args.model]
    leaderboard = []

    for task in tasks_to_run:
        X_train, X_test, y_train, y_test, groups_train, groups_test, feature_cols = load_train_test_datasets(
            args.train_data,
            args.test_data,
            task,
        )
        split_info = load_split_manifest()

        print("Preprocessing: z-normalization fit on 80% train split only")
        print("Tuning: subject-level CV inside train split")
        print("Final evaluation: fixed 20% holdout test split")
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
                name, X_train, y_train, groups_train, task, args
            )

            # recompute CV folds used (honor FAST_SEARCH_SETTINGS)
            fast = FAST_SEARCH_SETTINGS.get(name, {})
            cv_folds = int(fast.get("cv_folds", args.cv_folds) or args.cv_folds)
            cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=args.random_state)

            # compute per-fold CV metrics for the chosen estimator
            print("\nComputing per-fold CV metrics for best estimator...")
            cv_metrics = _compute_cv_metrics(best_pipeline, X_train, y_train, groups_train, cv)

            # summarize cv metrics: mean, std, per_fold
            cv_summary = {}
            for key, values in cv_metrics.items():
                if key == "confusion_matrix":
                    # convert list of matrices to array with padding if needed
                    mats = [np.array(m) for m in values]
                    max_dim = max(m.shape[0] for m in mats)
                    stacked = np.zeros((len(mats), max_dim, max_dim), dtype=float)
                    for i, m in enumerate(mats):
                        stacked[i, : m.shape[0], : m.shape[1]] = m
                    mean_mat = stacked.mean(axis=0).tolist()
                    std_mat = stacked.std(axis=0).tolist()
                    cv_summary[key] = {
                        "per_fold": [m.tolist() for m in mats],
                        "mean_matrix": mean_mat,
                        "std_matrix": std_mat,
                    }
                else:
                    # numeric metrics; some entries (like AUC) may be None
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

            # print CV summary (show mean+std and mean-std values)
            print("\nCV metrics (train-split folds):")
            for metric in ["auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1_score"]:
                m = cv_summary.get(metric, {})
                if m and m["mean"] is not None:
                    mean = m["mean"]
                    std = m.get("std") if m.get("std") is not None else 0.0
                    plus = mean + std
                    minus = mean - std
                    # output the two computed numbers (mean+std, mean-std)
                    print(f"- {metric}: {plus:.4f}, {minus:.4f}")
                else:
                    print(f"- {metric}: N/A")

            train_metrics = evaluate_pipeline(best_pipeline, X_train, y_train, task)
            test_metrics = evaluate_pipeline(best_pipeline, X_test, y_test, task)

            # Group test events by subject_id and evaluate at patient (subject) level
            try:
                df_test = X_test.copy()
                # ensure index-aligned labels and groups
                df_test = pd.DataFrame(df_test)
                df_test["subject_id"] = pd.Series(groups_test)
                df_test["label"] = pd.Series(y_test.values)

                grouped = df_test.groupby("subject_id")
                grouped_X_test = grouped[feature_cols].mean()

                # For subject-level label, take mode (most common event label); fallback to first
                def _mode_or_first(s):
                    m = s.mode()
                    return int(m.iloc[0]) if not m.empty else int(s.iloc[0])

                grouped_y_test = grouped["label"].agg(_mode_or_first)
                grouped_y_test = grouped_y_test.astype(int)

                grouped_test_metrics = evaluate_pipeline(best_pipeline, grouped_X_test, grouped_y_test, task)
                grouped_test_feature_stats = summarize_feature_statistics(grouped_X_test, feature_cols)
            except Exception:
                grouped_test_metrics = None
                grouped_test_feature_stats = []

            train_feature_stats = summarize_feature_statistics(X_train, feature_cols)
            test_feature_stats = summarize_feature_statistics(X_test, feature_cols)

            print(f"\nBest train-split CV balanced accuracy: {cv_score:.4f}")
            print(f"Best params: {json.dumps(best_params, indent=2)}")
            print_feature_statistics(train_feature_stats, label=f"{name.upper()} train features")
            print_feature_statistics(test_feature_stats, label=f"{name.upper()} test features")
            print_metrics(name.upper(), task, train_metrics, label="Train (80%)")
            print_metrics(name.upper(), task, test_metrics, label="Test (20%)")
            gap = train_metrics["balanced_accuracy"] - test_metrics["balanced_accuracy"]
            print(f"\nTrain-test balanced accuracy gap: {gap:.4f}")

            output_path = args.output_dir / f"{name}_{task}_tuned.joblib"
            joblib.dump(
                {
                    "model": best_pipeline,
                    "task": task,
                    "model_name": name,
                    "features": feature_cols,
                    "best_params": best_params,
                    "validation": "fixed_80_20_subject_split",
                    "normalization": "train_fit_zscore",
                    "cv_balanced_accuracy": cv_score,
                    "cv_metrics_summary": cv_summary,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "grouped_test_metrics": grouped_test_metrics,
                    "feature_statistics": {
                        "train": train_feature_stats,
                        "test": test_feature_stats,
                        "grouped_test": grouped_test_feature_stats,
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
                        "validation": "fixed_80_20_subject_split",
                        "normalization": "train_fit_zscore",
                        "best_params": best_params,
                        "cv_balanced_accuracy": cv_score,
                        "cv_metrics_summary": cv_summary,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "grouped_test_metrics": grouped_test_metrics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            # If grouped test metrics were computed, save a grouped-model artifact and metrics
            try:
                if grouped_test_metrics is not None:
                    grouped_output_dir = PROJECT_ROOT / "models" / "grouped_models_sudden"
                    grouped_output_dir.mkdir(parents=True, exist_ok=True)

                    grouped_artifact_path = grouped_output_dir / f"{name}_{task}_grouped_model.joblib"
                    joblib.dump(
                        {
                            "model": best_pipeline,
                            "task": task,
                            "model_name": name,
                            "features": feature_cols,
                            "best_params": best_params,
                            "grouped_test_metrics": grouped_test_metrics,
                            "grouped_test_feature_statistics": grouped_test_feature_stats,
                        },
                        grouped_artifact_path,
                    )

                    grouped_metrics_path = grouped_output_dir / f"{name}_{task}_grouped_metrics.json"
                    grouped_metrics_path.write_text(
                        json.dumps(
                            {
                                "model": name,
                                "task": task,
                                "grouped_test_metrics": grouped_test_metrics,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                    print(f"Saved grouped model artifact -> {grouped_artifact_path}")
                    print(f"Saved grouped metrics -> {grouped_metrics_path}")
            except Exception:
                print("Warning: failed to save grouped model artifacts.")

            leaderboard.append(
                {
                    "model": name,
                    "task": task,
                    "output_path": str(output_path),
                    "cv_balanced_accuracy": cv_score,
                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "test_f1_score": test_metrics["f1_score"],
                    "test_auc": test_metrics["auc"],
                }
            )

    if not leaderboard:
        raise RuntimeError("No models were trained.")

    leaderboard.sort(key=lambda row: (row["test_balanced_accuracy"], row["cv_balanced_accuracy"]), reverse=True)
    print("\n" + "=" * 80)
    print("LEADERBOARD (by 20% holdout balanced accuracy)")
    print("=" * 80)
    for rank, row in enumerate(leaderboard, start=1):
        auc = row["test_auc"]
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(
            f"{rank}. {row['model']} ({row['task']}): "
            f"cv_bal_acc={row['cv_balanced_accuracy']:.4f}, "
            f"holdout_bal_acc={row['test_balanced_accuracy']:.4f}, "
            f"f1={row['test_f1_score']:.4f}, "
            f"auc={auc_str}"
        )

    # select best model per task (binary and multiclass) by holdout balanced accuracy
    best_models_info = {}
    for task_name in ["binary", "multiclass"]:
        candidates = [row for row in leaderboard if row["task"] == task_name]
        if not candidates:
            continue
        # pick the top candidate for this task
        candidates.sort(key=lambda r: (r["test_balanced_accuracy"], r["cv_balanced_accuracy"]), reverse=True)
        best = candidates[0]
        artifact = joblib.load(best["output_path"])
        artifact["selected_by"] = "holdout_balanced_accuracy"
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
            "test_balanced_accuracy": best["test_balanced_accuracy"],
            "test_f1_score": best["test_f1_score"],
            "test_auc": best["test_auc"],
        }

        print(f"Saved best {task_name} model -> {model_path}")

    # write combined best-model info
    best_metrics_path = args.best_model_dir / "best_model_info.json"
    best_metrics_path.write_text(json.dumps(best_models_info, indent=2), encoding="utf-8")
    print(f"\nSaved best-model info -> {best_metrics_path}")


if __name__ == "__main__":
    main()
