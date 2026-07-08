"""
Generate 6 comparison tables (as PNG images) for model performance metrics
- 3 model types: tuned, grouped_tuned, grouped_sudden
- 2 tasks: binary, multiclass
- 5 algorithms per table: SVM, Random Forest, Logistic Regression, XGBoost, CatBoost
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

# Configuration
ALGORITHMS = ["svm", "rf", "lr", "xgb", "catboost"]
TASKS = ["binary", "multiclass"]
MODEL_TYPES = {
    "tuned": "models/tuned",
    "grouped_tuned": "models/grouped_tuned",
    "grouped_sudden": "models/grouped_models_sudden"
}
METRIC_KEYS = {
    # most metric files use the `test_metrics` key for holdout results
    "tuned": "test_metrics",
    "grouped_tuned": "test_metrics",
    "grouped_sudden": "test_metrics"
}
OUTPUT_DIR = Path("model_comparison_tables")

# Key metrics to display
METRICS_TO_DISPLAY = [
    "balanced_accuracy",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc" if "binary" else "None"  # Will be handled separately
]

BINARY_METRICS = [
    "balanced_accuracy",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc"
]

MULTICLASS_METRICS = [
    "balanced_accuracy",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc"
]


def load_metrics(model_type, algorithm, task):
    """Load metrics from model JSON file"""
    try:
        model_dir = MODEL_TYPES[model_type]
        algo = algorithm.lower()

        # possible filenames to try (grouped files include the '_grouped' suffix)
        candidates = [
            Path(model_dir) / f"{algo}_{task}_grouped_metrics.json",
            Path(model_dir) / f"{algo}_{task}_metrics.json",
            Path(model_dir) / f"{algo}_metrics.json",
        ]

        data = None
        for p in candidates:
            if p.exists():
                with open(p, 'r') as f:
                    data = json.load(f)
                break

        if data is None:
            print(f"Metrics file not found for {model_type} {algorithm} {task} (looked in {model_dir})")
            return None

        return data
    except Exception as e:
        print(f"Error loading {model_type} {algorithm} {task}: {e}")
        return None


def create_comparison_table(model_type, task):
    """Create a comparison table for a specific model type and task"""
    
    # Determine which metrics to use
    metrics = BINARY_METRICS if task == "binary" else MULTICLASS_METRICS
    
    # Load metrics for all algorithms
    table_data = []
    for algo in ALGORITHMS:
        metrics_dict = load_metrics(model_type, algo, task)
        
        if metrics_dict is None:
            continue
        
        row = {"Algorithm": algo.upper()}
        
        for metric in metrics:
            found = False
            # Search order: cv_metrics_summary first (since we want mean +- sd), then test_metrics, then train_metrics
            for k in ['cv_metrics_summary', 'test_metrics', 'train_metrics']:
                if k in metrics_dict and isinstance(metrics_dict[k], dict):
                    # handle recall/sensitivity mapping
                    metric_key = metric
                    if metric == "recall" and "recall" not in metrics_dict[k] and "sensitivity" in metrics_dict[k]:
                        metric_key = "sensitivity"
                    elif metric == "sensitivity" and "sensitivity" not in metrics_dict[k] and "recall" in metrics_dict[k]:
                        metric_key = "recall"
                        
                    if metric_key in metrics_dict[k]:
                        v = metrics_dict[k][metric_key]
                        if isinstance(v, dict) and 'mean' in v:
                            mean_val = v['mean']
                            std_val = v.get('std', 0.0)
                            if std_val is None:
                                std_val = 0.0
                            row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                        else:
                            row[metric] = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                        found = True
                        break
            if not found:
                row[metric] = "N/A"
        
        table_data.append(row)
    
    return pd.DataFrame(table_data)


def render_table_as_image(df, title, filename):
    """Render a dataframe as a table image and save as PNG"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.12] + [0.14] * (len(df.columns) - 1)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows with alternating colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Save as PNG
    output_file = OUTPUT_DIR / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(output_file)


def main():
    """Main function to generate all 6 comparison tables"""
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("GENERATING MODEL COMPARISON TABLES")
    print("=" * 80)
    
    table_count = 0
    
    for model_type in MODEL_TYPES.keys():
        for task in TASKS:
            table_count += 1
            
            # Create title
            model_display = model_type.replace("_", " ").title()
            task_display = task.title()
            title = f"Model Performance Comparison - {model_display} ({task_display})"
            
            # Create filename
            filename = f"comparison_{model_type}_{task}.png"
            
            print(f"\n[Table {table_count}/6] {model_display} - {task_display}")
            print(f"  Creating comparison table...")
            
            # Create dataframe
            df = create_comparison_table(model_type, task)
            
            if df.empty:
                print(f"  [WARNING] No data found for this configuration")
                continue
            
            # Render and save
            output_file = render_table_as_image(df, title, filename)
            print(f"  [SUCCESS] Saved: {filename}")
            print(f"  Algorithms: {', '.join(df['Algorithm'].str.lower().tolist())}")
            print(f"  Metrics: {len(df.columns) - 1}")
    
    print("\n" + "=" * 80)
    print(f"[SUCCESS] Generated {table_count} comparison tables in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
