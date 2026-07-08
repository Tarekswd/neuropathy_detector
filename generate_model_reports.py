"""
Generate comprehensive reports for all 30 models (tuned, grouped, grouped_sudden).
Reports include metrics (mean±std in decimal format), hyperparameters used in GridSearchCV, 
and best hyperparameters found.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import hyperparameter distributions from tune_models
from models.tune_models import PARAM_DISTRIBUTIONS

# Define paths
TUNED_DIR = PROJECT_ROOT / "models" / "tuned"
GROUPED_TUNED_DIR = PROJECT_ROOT / "models" / "grouped_tuned"
GROUPED_SUDDEN_DIR = PROJECT_ROOT / "models" / "grouped_models_sudden"
REPORTS_DIR = PROJECT_ROOT / "model_reports"

# Models and tasks
MODELS = ["svm", "rf", "lr", "xgb", "catboost"]
TASKS = ["binary", "multiclass"]

# Model type configurations
MODEL_CONFIGS = {
    "tuned": {
        "dir": TUNED_DIR,
        "name": "Ungrouped (Event-Level)",
        "metric_suffix": "_metrics.json"
    },
    "grouped_tuned": {
        "dir": GROUPED_TUNED_DIR,
        "name": "Grouped Patient-Level",
        "metric_suffix": "_grouped_metrics.json"
    },
    "grouped_sudden": {
        "dir": TUNED_DIR,  # Read from tuned models directory
        "name": "Grouped Sudden (Tuned Models on Grouped Data)",
        "metric_suffix": "_metrics.json",
        "metric_key": "grouped_test_metrics"  # Use grouped_test_metrics from tuned models
    }
}


def format_hyperparams(param_dict):
    """Format hyperparameters for display."""
    if not param_dict:
        return "None"
    
    formatted = []
    for key, value in sorted(param_dict.items()):
        # Remove 'model__' prefix for readability
        clean_key = key.replace("model__", "")
        formatted.append(f"{clean_key}={value}")
    return ", ".join(formatted)


def generate_model_report(model_type, model_name, task_name, metrics_data):
    """Generate a formatted report for a single model."""
    
    config = MODEL_CONFIGS[model_type]
    
    # Extract metrics - handle different formats based on model type
    best_params = metrics_data.get("best_params", {})
    
    # For grouped_sudden, extract grouped_test_metrics from tuned model metrics
    if model_type == "grouped_sudden" and "metric_key" in config:
        # Use the specified metric key (grouped_test_metrics) from tuned model
        test_metrics = metrics_data.get(config["metric_key"], {})
        cv_summary = {}  # No CV metrics for grouped_sudden
    else:
        # All model types have cv_metrics_summary
        cv_summary = metrics_data.get("cv_metrics_summary", {})
        
        # Extract test metrics (name varies by model type)
        if "grouped_test_metrics" in metrics_data:
            test_metrics = metrics_data.get("grouped_test_metrics", {})
        elif "test_metrics" in metrics_data:
            test_metrics = metrics_data.get("test_metrics", {})
        else:
            test_metrics = {}
    
    # Extract confusion matrix (name varies by model type)
    confusion_matrix_data = metrics_data.get("confusion_matrix", {})
    
    # Get hyperparameter search space for this model/task
    param_space = PARAM_DISTRIBUTIONS.get(model_name, {})
    if isinstance(param_space, dict):
        if task_name == "binary" and "binary" in param_space:
            param_space = param_space["binary"]
        elif task_name == "multiclass" and "multiclass" in param_space:
            param_space = param_space["multiclass"]
        # else keep the full dict or empty
    
    report = []
    report.append("=" * 100)
    report.append(f"MODEL REPORT: {model_name.upper()} - {task_name.upper()}")
    report.append("=" * 100)
    report.append(f"Model Type: {config['name']}")
    report.append("")
    
    # ---- Hyperparameter Search Space (skip for grouped_sudden) ----
    if model_type != "grouped_sudden":
        report.append("HYPERPARAMETER SEARCH SPACE (GridSearchCV)")
        report.append("-" * 100)
        if param_space:
            for param, values in sorted(param_space.items()):
                clean_param = param.replace("model__", "")
                report.append(f"  {clean_param}: {values}")
        else:
            report.append("  No hyperparameter tuning performed")
        report.append("")
    
    # ---- Best Hyperparameters ----
    if model_type != "grouped_sudden":
        report.append("BEST HYPERPARAMETERS (From GridSearchCV)")
        report.append("-" * 100)
        if best_params:
            if isinstance(best_params, dict):
                for param, value in sorted(best_params.items()):
                    clean_param = param.replace("model__", "")
                    report.append(f"  {clean_param} = {value}")
            elif isinstance(best_params, str):
                report.append(f"  {best_params}")
        else:
            report.append("  No best parameters found (model may not have been tuned)")
        report.append("")
    else:
        # For grouped_sudden, show that it uses best hyperparameters from ungrouped tuning
        report.append("CONFIGURATION")
        report.append("-" * 100)
        if best_params and isinstance(best_params, dict):
            report.append("  Using best hyperparameters from ungrouped (tuned) model:")
            for param, value in sorted(best_params.items()):
                clean_param = param.replace("model__", "")
                report.append(f"    {clean_param} = {value}")
        else:
            report.append("  Using best hyperparameters from ungrouped (tuned) model")
        report.append("")
    
    # Check if this has CV metrics or just test metrics
    if cv_summary:
        # ---- Performance Metrics (Decimal Format: mean ± std) ----
        report.append("CROSS-VALIDATION PERFORMANCE METRICS (mean ± std)")
        report.append("-" * 100)
        
        metric_names = ["balanced_accuracy", "accuracy", "auc", "precision", "recall", "f1_score", "sensitivity", "specificity"]
        
        for metric in metric_names:
            if metric in cv_summary:
                data = cv_summary[metric]
                mean = data.get("mean", None)
                std = data.get("std", None)
                
                if mean is not None and std is not None:
                    # Format as decimal: 0.75 ± 0.05
                    report.append(f"  {metric:20s}: {mean:.4f} ± {std:.4f}")
                elif mean is not None:
                    report.append(f"  {metric:20s}: {mean:.4f} ± N/A")
        report.append("")
        
        # ---- Per-Fold Breakdown ----
        report.append("PER-FOLD BREAKDOWN")
        report.append("-" * 100)
        
        # Show per-fold values for main metrics
        main_metrics = ["balanced_accuracy", "accuracy", "auc"]
        
        for metric in main_metrics:
            if metric in cv_summary and "per_fold" in cv_summary[metric]:
                per_fold = cv_summary[metric]["per_fold"]
                report.append(f"  {metric}:")
                for fold_idx, fold_value in enumerate(per_fold, 1):
                    if fold_value is not None:
                        report.append(f"    Fold {fold_idx}: {fold_value:.4f}")
                    else:
                        report.append(f"    Fold {fold_idx}: N/A")
        report.append("")
        
        # ---- Summary Statistics ----
        if model_type != "grouped_sudden":
            report.append("SUMMARY STATISTICS")
            report.append("-" * 100)
            
            # Compute range and other stats from per-fold values if available
            if "balanced_accuracy" in cv_summary:
                ba_data = cv_summary["balanced_accuracy"]
                if "per_fold" in ba_data:
                    folds = ba_data["per_fold"]
                    report.append(f"  Balanced Accuracy Range: {min(folds):.4f} to {max(folds):.4f}")
                    report.append(f"  Number of Folds: {len(folds)}")
            report.append("")
    elif test_metrics and model_type == "grouped_sudden":
        # For grouped_sudden, show the grouped test metrics
        report.append("GROUPED TEST SET PERFORMANCE METRICS")
        report.append("-" * 100)
        
        metric_names = ["balanced_accuracy", "accuracy", "auc", "precision", "recall", "f1_score", "sensitivity", "specificity"]
        
        for metric in metric_names:
            if metric in test_metrics:
                value = test_metrics.get(metric)
                if value is not None:
                    report.append(f"  {metric:20s}: {value:.4f}")
                else:
                    report.append(f"  {metric:20s}: N/A")
        report.append("")
    
    # Show confusion matrix for all models (at end)
    if confusion_matrix_data:
        report.append("CONFUSION MATRIX")
        report.append("-" * 100)
        labels = confusion_matrix_data.get("labels", [])
        matrix = confusion_matrix_data.get("matrix", [])
        report.append(f"  Labels: {labels}")
        for i, row in enumerate(matrix):
            report.append(f"  Row {i}: {row}")
        report.append("")
    elif "confusion_matrix" in test_metrics:
        report.append("CONFUSION MATRIX")
        report.append("-" * 100)
        cm = test_metrics["confusion_matrix"]
        labels = test_metrics.get("confusion_matrix_labels", list(range(len(cm))))
        report.append(f"  Labels: {labels}")
        for i, row in enumerate(cm):
            report.append(f"  Row {i}: {row}")
        report.append("")
    
    else:
        report.append("NO METRICS AVAILABLE")
        report.append("-" * 100)
        report.append("  Neither cross-validation nor test set metrics were found for this model.")
        report.append("")
    
    return "\n".join(report)


def main():
    """Generate reports for all 30 models."""
    
    # Create reports directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating reports in: {REPORTS_DIR}")
    print()
    
    report_count = 0
    
    # Iterate through all model types
    for model_type, config in MODEL_CONFIGS.items():
        model_dir = config["dir"]
        
        if not model_dir.exists():
            print(f"⚠ Skipping {model_type}: directory not found ({model_dir})")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {config['name']} Models ({model_type})")
        print(f"{'='*80}")
        
        # Iterate through all models and tasks
        for model_name in MODELS:
            for task_name in TASKS:
                # Construct metrics filename
                metrics_filename = f"{model_name}_{task_name}{config['metric_suffix']}"
                metrics_path = model_dir / metrics_filename
                
                if not metrics_path.exists():
                    print(f"  ⚠ {model_name} ({task_name}): metrics file not found")
                    continue
                
                # Read metrics
                with open(metrics_path, "r") as f:
                    metrics_data = json.load(f)
                
                # Generate report
                report_text = generate_model_report(model_type, model_name, task_name, metrics_data)
                
                # Save report
                report_filename = f"{model_type}_{model_name}_{task_name}_report.txt"
                report_path = REPORTS_DIR / report_filename
                
                with open(report_path, "w") as f:
                    f.write(report_text)
                
                print(f"  ✓ {model_name:10s} ({task_name:10s}) -> {report_filename}")
                report_count += 1
    
    print(f"\n{'='*80}")
    print(f"✓ Generated {report_count} reports in: {REPORTS_DIR}")
    print(f"{'='*80}")
    
    # Create a summary index file
    create_index_file(report_count)


def create_index_file(total_reports):
    """Create an index file listing all reports."""
    
    index_path = REPORTS_DIR / "INDEX.md"
    
    lines = []
    lines.append("# Model Reports Index")
    lines.append("")
    lines.append(f"**Total Reports:** {total_reports}")
    lines.append("")
    
    # Group by model type
    for model_type, config in MODEL_CONFIGS.items():
        lines.append(f"## {config['name']} ({model_type})")
        lines.append("")
        
        report_files = sorted(REPORTS_DIR.glob(f"{model_type}_*_report.txt"))
        if report_files:
            for report_file in report_files:
                lines.append(f"- [{report_file.name}]({report_file.name})")
        else:
            lines.append("  (No reports found)")
        
        lines.append("")
    
    with open(index_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"✓ Index file created: {index_path}")


if __name__ == "__main__":
    main()
