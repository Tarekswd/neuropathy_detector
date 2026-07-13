"""
Evaluate tuned (ungrouped) models on grouped patient-level test data.
This creates the "grouped_sudden" model type: tuned models tested on grouped data.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)


def compute_metrics(y_true, y_pred, y_pred_proba, task):
    """Compute 8 evaluation metrics."""
    metrics = {}
    
    try:
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    except:
        metrics['balanced_accuracy'] = None
    
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except:
        metrics['accuracy'] = None
    
    try:
        if task == "binary" and y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['auc'] = None
    except:
        metrics['auc'] = None
    
    try:
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    except:
        metrics['precision'] = None
    
    try:
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    except:
        metrics['recall'] = None
    
    try:
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    except:
        metrics['f1_score'] = None
    
    # Sensitivity and Specificity (for binary only)
    if task == "binary":
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        except:
            metrics['sensitivity'] = None
            metrics['specificity'] = None
    else:
        metrics['sensitivity'] = None
        metrics['specificity'] = None
    
    return metrics


def main():
    print("Loading grouped patient-level data...")
    
    train_df = pd.read_csv('grouped_patient_footprints/grouped_patient_features_train.csv')
    test_df = pd.read_csv('grouped_patient_footprints/grouped_patient_features_test.csv')
    
    exclude_cols = {"group", "class_id", "subject_id", "event_count", "event_ids", "source_folders"}
    feature_columns = [col for col in train_df.columns if col not in exclude_cols]
    
    grouped_train_data = train_df[feature_columns].to_numpy(dtype=float)
    grouped_test_data = test_df[feature_columns].to_numpy(dtype=float)
    
    grouped_train_labels_binary = (train_df["class_id"].to_numpy() >= 2).astype(int)
    grouped_train_labels_multiclass = train_df["class_id"].to_numpy()
    grouped_test_labels_binary = (test_df["class_id"].to_numpy() >= 2).astype(int)
    grouped_test_labels_multiclass = test_df["class_id"].to_numpy()
    
    print(f"  Grouped train data shape: {grouped_train_data.shape}")
    print(f"  Grouped test data shape: {grouped_test_data.shape}")
    
    # Select top 20 features by variance
    all_features = np.vstack([grouped_train_data, grouped_test_data])
    feature_variance = np.var(all_features, axis=0)
    top_k_indices = np.argsort(feature_variance)[-20:]
    top_k_indices = np.sort(top_k_indices)
    
    grouped_train_data_selected = grouped_train_data[:, top_k_indices]
    grouped_test_data_selected = grouped_test_data[:, top_k_indices]
    
    print(f"  Selected top 20 features")
    print(f"  Grouped train data (selected): {grouped_train_data_selected.shape}")
    print(f"  Grouped test data (selected): {grouped_test_data_selected.shape}")
    
    # Apply StandardScaler normalization
    scaler = StandardScaler()
    grouped_train_data_scaled = scaler.fit_transform(grouped_train_data_selected)
    grouped_test_data_scaled = scaler.transform(grouped_test_data_selected)
    
    # Create output directory
    output_dir = 'models/grouped_models_sudden'
    os.makedirs(output_dir, exist_ok=True)
    
    # Model configurations
    models_config = [
        ('svm', 'binary'),
        ('svm', 'multiclass'),
        ('rf', 'binary'),
        ('rf', 'multiclass'),
        ('lr', 'binary'),
        ('lr', 'multiclass'),
        ('xgb', 'binary'),
        ('xgb', 'multiclass'),
        ('catboost', 'binary'),
        ('catboost', 'multiclass'),
    ]
    
    leaderboard = []
    all_results = []
    
    for algorithm, task in models_config:
        print(f"\n{'='*80}")
        print(f"Processing: {algorithm.upper()} - {task.upper()}")
        print(f"{'='*80}")
        
        # Load tuned model
        model_path = f'models/tuned/{algorithm}_{task}_tuned_model.joblib'
        print(f"Loading tuned model from: {model_path}")
        
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            continue
        
        # Select appropriate labels
        if task == "binary":
            grouped_test_labels = grouped_test_labels_binary
        else:
            grouped_test_labels = grouped_test_labels_multiclass
        
        # Predict on grouped test data
        print(f"Predicting on grouped test data...")
        y_pred = model.predict(grouped_test_data_scaled)
        
        try:
            y_pred_proba = model.predict_proba(grouped_test_data_scaled)
        except:
            y_pred_proba = None
        
        # Compute metrics
        test_metrics = compute_metrics(grouped_test_labels, y_pred, y_pred_proba, task)
        
        # Compute confusion matrix
        cm = confusion_matrix(grouped_test_labels, y_pred)
        cm_labels = sorted(np.unique(grouped_test_labels))
        
        # Create metrics data structure
        metrics_data = {
            'model': algorithm,
            'task': task,
            'model_type': 'grouped_sudden',
            'description': 'Tuned (ungrouped) models tested on grouped patient-level data',
            'best_params': 'tuned_from_ungrouped_data',
            'test_metrics': test_metrics,
            'confusion_matrix': {
                'labels': cm_labels,
                'matrix': cm.tolist()
            }
        }
        
        # Save metrics
        metrics_path = f'{output_dir}/{algorithm}_{task}_grouped_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        all_results.append(metrics_data)
        print(f"  ✓ Saved metrics JSON: {metrics_path}")
        
        # Add to leaderboard
        leaderboard.append({
            'model': algorithm,
            'task': task,
            'balanced_accuracy': test_metrics.get('balanced_accuracy'),
            'accuracy': test_metrics.get('accuracy'),
            'f1_score': test_metrics.get('f1_score'),
            'auc': test_metrics.get('auc')
        })
    
    # Save leaderboard
    leaderboard_path = f'{output_dir}/leaderboard.json'
    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    # Save consolidated JSON summary
    summary_path = f'{output_dir}/all_grouped_results.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Saved leaderboard: {leaderboard_path}")
    print(f"✓ Saved consolidated results: {summary_path}")
    
    print(f"\n{'='*80}")
    print("✓ Completed evaluation of tuned models on grouped test data")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
