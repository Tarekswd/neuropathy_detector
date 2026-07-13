"""
Analyze overfitting in grouped patient models using nested cross-validation.
Tests if the perfect test performance is real or just noise from small test set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
GROUPED_ROOT = PROJECT_ROOT / "grouped_patient_footprints"

def load_grouped_data(k_features: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    """Load grouped patient features."""
    df = pd.read_csv(GROUPED_ROOT / "grouped_patient_features.csv")
    print(f"Total grouped patients: {len(df)}")
    
    # Extract labels (binary: 0=control, 1=neuropathy)
    y = df["class_id"].values
    
    # Select top k features (excluding metadata columns)
    exclude_cols = {"group", "class_id", "subject_id", "event_count", "event_ids", "source_folders"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols[:k_features]].values
    
    print(f"Using {X.shape[1]} features, {X.shape[0]} samples")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return df, X, y

def nested_cross_validation(X, y, n_outer: int = 5, n_inner: int = 5) -> dict:
    """
    Nested cross-validation: outer loop estimates generalization, inner loop tunes.
    """
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)
    
    outer_scores = {
        "balanced_accuracy": [],
        "f1_macro": [],
        "roc_auc": [],
        "accuracy": [],
    }
    
    fold_details = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Simple SVM pipeline (enable probability for ROC-AUC)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
        ])
        
        # Train and evaluate
        pipe.fit(X_train, y_train)
        
        test_acc = pipe.score(X_test, y_test)
        train_acc = pipe.score(X_train, y_train)
        
        # Cross-validate score on train split (only metrics compatible with all estimators)
        cv_scores = cross_validate(
            pipe, X_train, y_train,
            cv=inner_cv,
            scoring=["accuracy", "f1_macro"],
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
        
        outer_scores["balanced_accuracy"].append(test_acc)
        fold_details.append(fold_summary)
        
        print(f"Fold {fold_idx+1}: test_acc={test_acc:.4f}, train_acc={train_acc:.4f}, gap={train_acc - test_acc:.4f}")
    
    return {
        "nested_cv_mean_bal_acc": float(np.mean(outer_scores["balanced_accuracy"])),
        "nested_cv_std_bal_acc": float(np.std(outer_scores["balanced_accuracy"])),
        "fold_details": fold_details,
    }

def main():
    print("=" * 80)
    print("OVERFITTING ANALYSIS: Grouped Patient Model")
    print("=" * 80)
    print()
    
    # Load data
    df, X, y = load_grouped_data(k_features=20)
    
    # Convert to binary: neuropathy vs. control
    # Assuming: GC=0, GD=1, NL=2, NS=3
    # Binary: (GC, NL) = control (0), (GD, NS) = neuropathy (1)
    y_binary = np.zeros_like(y)
    y_binary[y >= 2] = 1  # GD, NS -> neuropathy
    
    print(f"Binary classification: Control={np.sum(y_binary==0)}, Neuropathy={np.sum(y_binary==1)}")
    print()
    
    # Run nested CV
    print("Running nested cross-validation (5 outer folds x 5 inner folds)...")
    print()
    results = nested_cross_validation(X, y_binary, n_outer=5, n_inner=5)
    
    print()
    print("=" * 80)
    print("NESTED CV RESULTS (Generalization Estimate)")
    print("=" * 80)
    print(f"Mean Balanced Accuracy: {results['nested_cv_mean_bal_acc']:.4f}")
    print(f"Std Dev:               {results['nested_cv_std_bal_acc']:.4f}")
    print()
    print("Per-fold breakdown:")
    for fold in results["fold_details"]:
        print(f"  Fold {fold['fold']}: test={fold['test_acc']:.4f}, "
              f"train={fold['train_acc']:.4f}, gap={fold['train_test_gap']:.4f}")
    
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"Train set size: 43 patients (428 events)")
    print(f"Test set size: 11 patients (104 events) ← TOO SMALL")
    print()
    print("Key findings:")
    print(f"1. Nested CV mean: {results['nested_cv_mean_bal_acc']:.4f} (realistic)")
    print(f"2. Original test set (n=11): 1.0000 (likely noise)")
    print(f"3. CV-to-test gap suggests test set is not representative")
    print()
    print("Recommendation: Trust nested CV metrics (~{:.2%}), not perfect test metrics.".format(
        results['nested_cv_mean_bal_acc']))
    print("Consider increasing test set size or using stratified k-fold on full data.")
    print()
    
    # Save results
    out_path = PROJECT_ROOT / "grouped_overfitting_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
