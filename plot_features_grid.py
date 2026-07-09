"""Generate a grid of plots for all 20 biomechanical features to visualize their distributions."""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import (
    GROUP_COLORS,
    GROUP_ORDER,
    REGIONS,
    METRICS,
    PROJECT_ROOT,
    ensure_dir,
    setup_matplotlib,
)

FEATURES_CSV = PROJECT_ROOT / "ml_features" / "npy_features_raw.csv"
OUTPUT_DIR = PROJECT_ROOT / "graph_plots"
EXTRACTION_SCRIPT = PROJECT_ROOT / "extract_npy_features.py"


def check_and_extract_features() -> None:
    """Check if features CSV exists. If not, trigger extraction."""
    if not FEATURES_CSV.exists():
        print(f"[Info] Features file not found at: {FEATURES_CSV}")
        if EXTRACTION_SCRIPT.exists():
            print(f"[Info] Running feature extraction: {EXTRACTION_SCRIPT.name}...")
            try:
                subprocess.run([sys.executable, str(EXTRACTION_SCRIPT)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to run extraction script: {e}")
                sys.exit(1)
        else:
            print(f"[Error] Feature extraction script not found at {EXTRACTION_SCRIPT}")
            sys.exit(1)


def load_data() -> pd.DataFrame:
    """Load and prepare features dataset."""
    df = pd.read_csv(FEATURES_CSV)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    return df


def plot_features_grid(df: pd.DataFrame) -> None:
    """Create a 4x5 grid of boxplots (4 regions x 5 metrics) showing feature distributions."""
    # Setup premium aesthetics
    setup_matplotlib()
    
    # 4 regions (rows) x 5 metrics (columns)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=5,
        figsize=(24, 18),
        sharex=True,
        constrained_layout=True,
    )
    
    # Color palette
    colors = [GROUP_COLORS[g] for g in GROUP_ORDER]
    rng = np.random.default_rng(42)

    for row_idx, region in enumerate(REGIONS):
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            feature_name = f"{region}_{metric}"
            
            if feature_name not in df.columns:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                continue
                
            # Prepare data groups
            data_groups = []
            for g in GROUP_ORDER:
                data_g = df.loc[df["group"] == g, feature_name].dropna().astype(float).values
                data_groups.append(data_g)
                
            # Create boxplot
            bp = ax.boxplot(
                data_groups,
                tick_labels=GROUP_ORDER,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                widths=0.5,
                boxprops=dict(linewidth=1.2, edgecolor='#333333'),
                whiskerprops=dict(color='#666666', linewidth=1),
                capprops=dict(color='#666666', linewidth=1),
                medianprops=dict(color='#d62728', linewidth=1.5),
                meanprops=dict(color='#2ca02c', linestyle='--', linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='#999999', markersize=4, markeredgecolor='none', alpha=0.5),
            )
            
            # Fill boxes with semi-transparent versions of group colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                
            # Add jittered raw data points in background
            for i, data_g in enumerate(data_groups, start=1):
                if len(data_g) > 0:
                    x_jitter = rng.normal(i, 0.08, size=len(data_g))
                    ax.scatter(
                        x_jitter,
                        data_g,
                        color=colors[i - 1],
                        s=10,
                        alpha=0.4,
                        edgecolors='none',
                        zorder=1
                    )
            
            # Formatting
            clean_title = f"{region.title()} - {metric.replace('_', ' ').title()}"
            ax.set_title(clean_title, fontsize=12, fontweight='semibold', pad=10)
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            
            # Remove top/right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            
            # Adjust y-label only on the first column to save space
            if col_idx == 0:
                ax.set_ylabel("Value", fontsize=10, fontweight='medium')
                
            # Adjust x-label only on the last row
            if row_idx == 3:
                ax.set_xlabel("Clinical Group", fontsize=10, fontweight='medium')

    # Add a main title for the whole grid
    fig.suptitle(
        "Distribution of 20 Biomechanical Features across Clinical Groups\n"
        "(GC = Control, GD = Diabetic (No Neuropathy), NL = Neuropathy Low, NS = Neuropathy Severe)\n"
        "Solid red line: Median | Dashed green line: Mean",
        fontsize=18,
        fontweight='bold',
        y=1.02
    )

    # Save visualization
    ensure_dir(OUTPUT_DIR)
    output_path = OUTPUT_DIR / "features_grid_distribution.png"
    
    # Save with tight bbox to prevent clipping the suptitle
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Success] Saved feature grid distribution plot to: {output_path}")


def main() -> None:
    check_and_extract_features()
    df = load_data()
    plot_features_grid(df)


if __name__ == "__main__":
    main()
