"""
Generate 4 comparison tables (as PNG images) for model performance metrics
- 2 model types: tuned, grouped_tuned
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
    "grouped_tuned": "models/grouped_tuned"
}
METRIC_KEYS = {
    # most metric files use the `test_metrics` key for holdout results
    "tuned": "test_metrics",
    "grouped_tuned": "test_metrics"
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


def _best_row_index(df):
    """Return the 0-based data row index of the best model by balanced_accuracy (or f1_score fallback)."""
    for col in ["balanced_accuracy", "f1_score", "accuracy"]:
        if col not in df.columns:
            continue
        def _parse(val):
            # handles "0.9123 ± 0.0045" or "0.9123" or "N/A"
            try:
                return float(str(val).split("±")[0].strip())
            except (ValueError, AttributeError):
                return -1.0
        scores = df[col].apply(_parse)
        if scores.max() > 0:
            return int(scores.idxmax())
    return None


def render_table_as_image(df, title, filename):
    """Render a dataframe as a table image and save as PNG"""

    n_rows = len(df)
    fig_height = 0.45 * (n_rows + 1) + 0.6  # compact: ~0.45 in per row + title room
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')

    # Determine best model row (0-based index in df)
    best_idx = _best_row_index(df)

    # Build per-cell colours
    cell_colors = []
    for i in range(n_rows):
        is_best = (best_idx is not None and i == best_idx)
        row_colors = []
        for _ in range(len(df.columns)):
            if is_best:
                row_colors.append('#C0392B')
            elif i % 2 == 0:
                row_colors.append('#F2F2F2')
            else:
                row_colors.append('#E7E6E6')
        cell_colors.append(row_colors)

    header_colors = [['#4472C4'] * len(df.columns)]

    # Create table filling the full axes area (no blank space above/below)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
        colColours=header_colors[0],
        colWidths=[0.12] + [0.14] * (len(df.columns) - 1),
        bbox=[0, 0, 1, 1],   # fill the entire axes bounding box
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header text
    for j in range(len(df.columns)):
        table[(0, j)].set_text_props(weight='bold', color='white')

    # Style best-row text
    if best_idx is not None:
        for j in range(len(df.columns)):
            table[(best_idx + 1, j)].set_text_props(weight='bold', color='white')

    plt.title(title, fontsize=13, fontweight='bold', pad=6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)

    # Save as PNG
    output_file = OUTPUT_DIR / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
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
            
            print(f"\n[Table {table_count}/4] {model_display} - {task_display}")
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
