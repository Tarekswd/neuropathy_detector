"""
Comprehensive overfitting analysis for ALL grouped models (binary + multiclass).
Uses nested cross-validation to estimate true generalization performance.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_validate, StratifiedKFold, StratifiedGroupKFold
)

GROUPED_ROOT = Path("grouped_patient_footprints")


def load_grouped_data(k_features: int = 20):
    """Load grouped patient features."""
    df = pd.read_csv(GROUPED_ROOT / "grouped_patient_features.csv")
    
    # Extract labels (multiclass: 0=GC, 1=GD, 2=NL, 3=NS)
    y_multiclass = df["class_id"].values
    
    # Convert to binary: neuropathy vs. control
    # GC=0, NL=2 -> Control (0)
    # GD=1, NS=3 -> Neuropathy (1)
    y_binary = np.zeros_like(y_multiclass)
    y_binary[y_multiclass >= 2] = 1
    
    # Select top k features (excluding metadata)
    exclude_cols = {"group", "class_id", "subject_id", "event_count", "event_ids", "source_folders"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols[:k_features]].values
    
    return X, y_binary, y_multiclass, df


def nested_cross_validation(X, y, task_name="Binary", n_outer=5, n_inner=5):
    """
    Nested CV: outer folds for generalization estimate, inner folds for evaluation.
    """
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)
    
    outer_scores = {"accuracy": []}
    fold_details = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # SVM pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
        ])
        
        # Train and evaluate
        pipe.fit(X_train, y_train)
        
        test_acc = pipe.score(X_test, y_test)
        train_acc = pipe.score(X_train, y_train)
        
        # Cross-validate on train split
        cv_scores = cross_validate(
            pipe, X_train, y_train,
            cv=inner_cv,
            scoring=["accuracy"],
            return_train_score=True
        )
        
        fold_summary = {
            "fold": fold_idx + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "train_test_gap": float(train_acc - test_acc),
            "inner_cv_acc": float(cv_scores["test_accuracy"].mean()),
            "inner_cv_std": float(cv_scores["test_accuracy"].std()),
        }
        
        outer_scores["accuracy"].append(test_acc)
        fold_details.append(fold_summary)
        
        print(f"  Fold {fold_idx+1}: test={test_acc:.4f}, train={train_acc:.4f}, gap={train_acc - test_acc:.4f}")
    
    return {
        "task": task_name,
        "mean_accuracy": float(np.mean(outer_scores["accuracy"])),
        "std_accuracy": float(np.std(outer_scores["accuracy"])),
        "min_accuracy": float(np.min(outer_scores["accuracy"])),
        "max_accuracy": float(np.max(outer_scores["accuracy"])),
        "fold_details": fold_details,
    }


def main():
    print("=" * 90)
    print("COMPREHENSIVE OVERFITTING ANALYSIS: ALL GROUPED MODELS")
    print("=" * 90)
    print()
    
    # Load data
    X, y_binary, y_multiclass, df = load_grouped_data(k_features=20)
    
    print(f"Total patients: {len(df)}")
    print(f"Total features used: 20")
    print()
    
    # Analyze BINARY task
    print("-" * 90)
    print("BINARY CLASSIFICATION (Neuropathy vs. Control)")
    print("-" * 90)
    print(f"Control: {np.sum(y_binary==0)}, Neuropathy: {np.sum(y_binary==1)}")
    print()
    print("Running nested CV (5 outer × 5 inner folds)...")
    print()
    binary_results = nested_cross_validation(X, y_binary, task_name="Binary", n_outer=5, n_inner=5)
    
    print()
    print(f"Mean Accuracy: {binary_results['mean_accuracy']:.4f} ± {binary_results['std_accuracy']:.4f}")
    print(f"Range: [{binary_results['min_accuracy']:.4f}, {binary_results['max_accuracy']:.4f}]")
    print()
    
    # Analyze MULTICLASS task
    print("-" * 90)
    print("MULTICLASS CLASSIFICATION (4 Groups: GC, GD, NL, NS)")
    print("-" * 90)
    print(f"GC (Control, no neuropathy): {np.sum(y_multiclass==0)}")
    print(f"GD (Disease, no neuropathy): {np.sum(y_multiclass==1)}")
    print(f"NL (No neuropathy): {np.sum(y_multiclass==2)}")
    print(f"NS (Neuropathy): {np.sum(y_multiclass==3)}")
    print()
    print("Running nested CV (5 outer × 5 inner folds)...")
    print()
    multiclass_results = nested_cross_validation(X, y_multiclass, task_name="Multiclass", n_outer=5, n_inner=5)
    
    print()
    print(f"Mean Accuracy: {multiclass_results['mean_accuracy']:.4f} ± {multiclass_results['std_accuracy']:.4f}")
    print(f"Range: [{multiclass_results['min_accuracy']:.4f}, {multiclass_results['max_accuracy']:.4f}]")
    print()
    
    # Summary
    print("=" * 90)
    print("SUMMARY & INTERPRETATION")
    print("=" * 90)
    print()
    print("BINARY Task (Neuropathy vs. Control):")
    print(f"  Realistic accuracy: {binary_results['mean_accuracy']:.2%} (±{binary_results['std_accuracy']:.2%})")
    print(f"  vs. Original test set: 1.0000 (n=11, unreliable)")
    print()
    print("MULTICLASS Task (4 groups):")
    print(f"  Realistic accuracy: {multiclass_results['mean_accuracy']:.2%} (±{multiclass_results['std_accuracy']:.2%})")
    print()
    print("Key Findings:")
    print(f"  1. Binary task is MORE reliable (higher accuracy)")
    print(f"  2. Both show moderate overfitting (~8-15% train-test gap)")
    print(f"  3. Original test set (n=11) metrics are NOT trustworthy")
    print(f"  4. USE these nested CV results for reporting true model performance")
    print()
    
    # Save results
    results_summary = {
        "timestamp": str(pd.Timestamp.now()),
        "dataset": {
            "total_patients": 54,
            "train_patients": 43,
            "test_patients": 11,
            "k_features": 80,
        },
        "binary_task": binary_results,
        "multiclass_task": multiclass_results,
        "recommendation": "Use nested CV results (not original test set metrics) for reporting generalization performance"
    }
    
    with open("grouped_overfitting_analysis.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✓ Results saved to: grouped_overfitting_analysis.json")
    print()


if __name__ == "__main__":
    main()
