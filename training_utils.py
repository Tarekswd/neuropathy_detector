from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from models.model_builders import MULTICLASS_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "ml_features" / "npy_features_train.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "ml_features" / "npy_features_test.csv"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "ml_features" / "npy_split_manifest.json"
BINARY_POSITIVE_GROUPS = {"NL", "NS"}
BINARY_POSITIVE_CLASS_IDS = {2, 3}
METADATA_COLUMNS = {"group", "class_id", "subject_id", "event_id", "source_folder"}


def configure_warnings() -> None:
    warnings.filterwarnings("ignore", message=".*probability.*deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Only one class is present in y_true.*", category=UserWarning)


def _metric_labels(y_true, y_pred, task: str) -> list[int]:
    if task == "binary":
        return [0, 1]
    observed = set(np.unique(y_true)) | set(np.unique(y_pred))
    return sorted(observed | set(MULTICLASS_LABELS))


def _score_matrix(pipeline, X) -> np.ndarray | None:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        if scores.ndim == 1:
            return scores
        return scores
    return None


def parse_args(script_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train {script_path.stem} classification model.")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5, help="Subject-level CV folds on the train split.")
    parser.add_argument("--k-features", type=int, default=None, help="Top K features (default: all features).")
    parser.add_argument("--model-output", type=Path, default=None)
    return parser.parse_args()


def _read_feature_frame(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = (
        df.drop(columns=list(METADATA_COLUMNS.intersection(df.columns)))
        .select_dtypes(include="number")
        .columns
        .tolist()
    )
    if not feature_columns:
        raise ValueError("No numeric features found.")
    return feature_columns


def _labels_from_frame(df: pd.DataFrame, task: str) -> pd.Series:
    if task == "binary":
        return df["group"].isin(BINARY_POSITIVE_GROUPS).astype(int)
    if task == "multiclass":
        return df["class_id"].astype(int)
    raise ValueError(f"Unknown task: {task}")


def load_train_test_datasets(train_path: Path, test_path: Path, task: str):
    train_df = _read_feature_frame(train_path)
    test_df = _read_feature_frame(test_path)

    for name, df in [("train", train_df), ("test", test_df)]:
        missing = {"group", "class_id", "subject_id"}.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {name} data: {missing}")

    feature_columns = _feature_columns(train_df)
    test_feature_columns = _feature_columns(test_df)
    if feature_columns != test_feature_columns:
        raise ValueError("Train and test feature columns do not match.")

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train = _labels_from_frame(train_df, task)
    y_test = _labels_from_frame(test_df, task)
    groups_train = train_df["subject_id"].astype(str).values
    groups_test = test_df["subject_id"].astype(str).values

    overlap = set(groups_train) & set(groups_test)
    if overlap:
        raise RuntimeError(f"Subject leakage detected between train and test files: {overlap}")

    return X_train, X_test, y_train, y_test, groups_train, groups_test, feature_columns


def load_split_manifest(path: Path = DEFAULT_SPLIT_MANIFEST) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _effective_k_features(k_features: int | None, n_features: int) -> int:
    if k_features is None or k_features <= 0:
        return n_features
    return min(k_features, n_features)


def summarize_feature_statistics(X, feature_columns: list[str] | None = None) -> list[dict]:
    if isinstance(X, pd.DataFrame):
        frame = X
    else:
        frame = pd.DataFrame(X)

    columns = feature_columns or frame.columns.tolist()
    summaries: list[dict] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=0))
        summaries.append(
            {
                "feature": str(column),
                "mean": mean_value,
                "std": std_value,
                "mean_std": f"{mean_value:.4f} ± {std_value:.4f}",
            }
        )
    return summaries


def print_feature_statistics(feature_summaries: list[dict], label: str = "Feature") -> None:
    print(f"\n{label} summary (mean ± std):")
    for summary in feature_summaries:
        print(f"  {summary['feature']}: {summary['mean_std']}")


def build_pipeline(model, k_features: int | None, n_features: int) -> Pipeline:
    k = _effective_k_features(k_features, n_features)
    steps = []
    if k < n_features:
        steps.append(("feature_selection", SelectKBest(f_classif, k=k)))
    steps.append(("model", model))
    return Pipeline(steps)


def compute_specificity(y_true, y_pred, labels: list[int], task: str) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if task == "binary":
        row_sum = cm[0, :].sum()
        return float(cm[0, 0] / row_sum) if row_sum > 0 else 0.0

    specificities: list[float] = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificities.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specificities))


def _binary_targets(y_true, y_pred, task: str) -> tuple[np.ndarray, np.ndarray]:
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)
    if task == "binary":
        return true_values.astype(int), pred_values.astype(int)
    return (
        np.isin(true_values.astype(int), list(BINARY_POSITIVE_CLASS_IDS)).astype(int),
        np.isin(pred_values.astype(int), list(BINARY_POSITIVE_CLASS_IDS)).astype(int),
    )


def compute_metrics(y_true, y_pred, y_scores=None, task: str = "binary"):
    average = "binary" if task == "binary" else "macro"
    labels = [int(label) for label in _metric_labels(y_true, y_pred, task)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    binary_y_true, binary_y_pred = _binary_targets(y_true, y_pred, task)
    binary_cm = confusion_matrix(binary_y_true, binary_y_pred, labels=[0, 1])

    results = {
        "auc": None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(
            recall_score(y_true, y_pred, labels=labels, average=average, zero_division=0)
        ),
        "specificity": compute_specificity(y_true, y_pred, labels, task),
        "precision": float(
            precision_score(y_true, y_pred, labels=labels, average=average, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "binary_confusion_matrix": binary_cm.tolist(),
        "binary_confusion_matrix_labels": ["non_neuropathy", "neuropathy"],
    }

    if y_scores is not None:
        try:
            if task == "binary":
                if isinstance(y_scores, np.ndarray) and y_scores.ndim == 2 and y_scores.shape[1] >= 2:
                    results["auc"] = float(roc_auc_score(y_true, y_scores[:, 1]))
                else:
                    results["auc"] = float(roc_auc_score(y_true, y_scores))
            else:
                y_bin = label_binarize(y_true, classes=MULTICLASS_LABELS)
                if y_scores.ndim == 1:
                    results["auc"] = None
                elif y_scores.shape[1] == len(MULTICLASS_LABELS):
                    results["auc"] = float(
                        roc_auc_score(y_bin, y_scores, average="macro", multi_class="ovr")
                    )
                else:
                    results["auc"] = float(
                        roc_auc_score(y_bin, y_scores, average="macro", multi_class="ovr")
                    )
        except ValueError:
            results["auc"] = None

    return results


def evaluate_pipeline(pipeline, X, y, task: str) -> dict:
    y_pred = pipeline.predict(X)
    y_scores = _score_matrix(pipeline, X)
    return compute_metrics(y, y_pred, y_scores, task=task)


def subject_cross_validation(
    pipeline_factory: Callable[[], Pipeline],
    X,
    y,
    groups,
    *,
    task: str,
    cv_folds: int,
    random_state: int,
) -> dict:
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_metrics: list[dict] = []

    for _, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        pipeline = pipeline_factory()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_metrics.append(evaluate_pipeline(pipeline, X.iloc[val_idx], y.iloc[val_idx], task))

    summary: dict = {"folds": cv_folds, "data": "train_split_only"}
    for key in [
        "auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1_score",
        "accuracy",
    ]:
        values = [m[key] for m in fold_metrics if m.get(key) is not None and not (isinstance(m[key], float) and np.isnan(m[key]))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    summary["last_confusion_matrix"] = fold_metrics[-1]["confusion_matrix"]
    return summary


def print_metrics(model_name: str, task: str, metrics: dict, label: str = "Test") -> None:
    print(f"\n{label} metrics ({model_name}, {task}):")
    print(f"  AUC:                 {metrics['auc']}")
    print(f"  Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
    print(f"  Sensitivity:         {metrics['sensitivity']:.4f}")
    print(f"  Specificity:         {metrics['specificity']:.4f}")
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  F1-score:            {metrics['f1_score']:.4f}")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print("  Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")
    print("  Binary Confusion Matrix (non-neuropathy vs neuropathy):")
    for row in metrics["binary_confusion_matrix"]:
        print(f"    {row}")


def print_cv_summary(cv_summary: dict) -> None:
    print("\nSubject-level cross-validation on train split:")
    for key in [
        "auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1_score",
        "accuracy",
    ]:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in cv_summary:
            print(f"  {key}: {cv_summary[mean_key]:.4f} +/- {cv_summary[std_key]:.4f}")


def train_and_save(
    *,
    script_path: Path,
    task: str,
    model_name: str,
    build_model: Callable[[int], object],
):
    configure_warnings()
    args = parse_args(script_path)
    X_train, X_test, y_train, y_test, groups_train, groups_test, feature_columns = load_train_test_datasets(
        args.train_data,
        args.test_data,
        task,
    )
    k_features = _effective_k_features(args.k_features, len(feature_columns))
    split_info = load_split_manifest()

    model = build_model(args.random_state)
    pipeline = build_pipeline(model, k_features, len(feature_columns))
    pipeline.fit(X_train, y_train)

    train_metrics = evaluate_pipeline(pipeline, X_train, y_train, task)
    test_metrics = evaluate_pipeline(pipeline, X_test, y_test, task)

    cv_summary = subject_cross_validation(
        pipeline_factory=lambda: build_pipeline(build_model(args.random_state), k_features, len(feature_columns)),
        X=X_train,
        y=y_train,
        groups=groups_train,
        task=task,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    train_feature_stats = summarize_feature_statistics(X_train, feature_columns)
    test_feature_stats = summarize_feature_statistics(X_test, feature_columns)

    print(f"\nModel: {model_name}")
    print(f"Task: {task}")
    print("Preprocessing: z-normalization fit on 80% train split only")
    print("Evaluation: fixed 20% subject-level holdout test split")
    print_feature_statistics(train_feature_stats, label="Train features")
    print_feature_statistics(test_feature_stats, label="Test features")
    if split_info:
        print(
            f"Split: {len(split_info.get('train_subjects', []))} train subjects "
            f"({split_info.get('train_events', len(X_train))} events) / "
            f"{len(split_info.get('test_subjects', []))} test subjects "
            f"({split_info.get('test_events', len(X_test))} events)"
        )

    print_metrics(model_name, task, train_metrics, label="Train (80%)")
    print_metrics(model_name, task, test_metrics, label="Test (20%)")
    gap = train_metrics["balanced_accuracy"] - test_metrics["balanced_accuracy"]
    print(f"\nTrain-test balanced accuracy gap: {gap:.4f}")
    print_cv_summary(cv_summary)

    model_output = args.model_output or script_path.with_name(f"{script_path.stem}_model.joblib")
    metrics_output = script_path.with_name(f"{script_path.stem}_metrics.json")
    model_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": pipeline,
            "task": task,
            "model_name": model_name,
            "feature_columns": feature_columns,
            "k_features": k_features,
            "validation": "fixed_80_20_subject_split",
            "normalization": "train_fit_zscore",
        },
        model_output,
    )

    output = {
        "task": task,
        "model_name": model_name,
        "train_data_path": str(args.train_data),
        "test_data_path": str(args.test_data),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_subjects": int(len(set(groups_train))),
        "test_subjects": int(len(set(groups_test))),
        "k_features": k_features,
        "validation": "fixed_80_20_subject_split",
        "normalization": "train_fit_zscore",
        "split_info": split_info,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_cv": cv_summary,
        "feature_statistics": {
            "train": train_feature_stats,
            "test": test_feature_stats,
        },
    }

    metrics_output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nSaved model -> {model_output}")
    print(f"Saved metrics -> {metrics_output}")
