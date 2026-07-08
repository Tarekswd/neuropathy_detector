"""Generate feature distribution graphs from ml_features/npy_features_raw.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import (
    GROUP_COLORS,
    GROUP_ORDER,
    METRICS,
    PROJECT_ROOT,
    REGIONS,
    ensure_dir,
    setup_matplotlib,
)

FEATURES_CSV = PROJECT_ROOT / "ml_features" / "npy_features_raw.csv"
OUTPUT_DIR = PROJECT_ROOT / "graph_plots"


def load_features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Run extract_npy_features.py first. Missing {FEATURES_CSV}"
        )

    df = pd.read_csv(FEATURES_CSV)
    df["group"] = pd.Categorical(
        df["group"],
        categories=GROUP_ORDER,
        ordered=True,
    )
    return df


def _scatter_by_group(
    ax,
    df: pd.DataFrame,
    column_name: str,
    title: str,
) -> None:

    rng = np.random.default_rng(42)

    for i, group_name in enumerate(GROUP_ORDER, start=1):

        values = (
            df.loc[df["group"] == group_name, column_name]
            .dropna()
            .astype(float)
            .values
        )

        if len(values) == 0:
            continue

        x = rng.normal(i, 0.06, len(values))

        # Individual participants
        ax.scatter(
            x,
            values,
            s=45,
            color=GROUP_COLORS[group_name],
            edgecolors="black",
            linewidths=0.4,
            alpha=0.75,
            zorder=2,
        )

        # Mean ± SD
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0

        ax.errorbar(
            i,
            mean,
            yerr=std,
            fmt="o",
            color="black",
            markersize=9,
            capsize=6,
            linewidth=2,
            zorder=3,
        )

    ax.set_xticks(range(1, len(GROUP_ORDER) + 1))
    ax.set_xticklabels(GROUP_ORDER)

    ax.set_xlabel("Clinical Group")
    ax.set_ylabel(column_name.replace("_", " ").title())
    ax.set_title(title.replace("_", " ").title())

    ax.grid(axis="y", linestyle="--", alpha=0.35)


def plot_feature_boxplots(df: pd.DataFrame) -> None:
    for region_name in REGIONS:
        for metric_name in METRICS:

            column_name = f"{region_name}_{metric_name}"

            if column_name not in df.columns:
                continue

            fig, ax = plt.subplots(
                figsize=(7, 5),
                constrained_layout=True,
            )

            _scatter_by_group(
                ax,
                df,
                column_name,
                column_name,
            )

            output_path = OUTPUT_DIR / f"{region_name}_{metric_name}.png"

            fig.savefig(output_path, dpi=300)
            plt.close(fig)

            print(f"Saved {output_path.name}")


def plot_descriptive_overview(df: pd.DataFrame) -> None:

    key_features = [
        "whole_total_pti",
        "whole_global_peak_pressure",
        "fore_total_pti",
        "heel_total_pti",
        "whole_contact_area",
        "whole_contact_duration_proxy",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
        constrained_layout=True,
    )

    for ax, feature in zip(axes.ravel(), key_features):

        if feature not in df.columns:
            continue

        _scatter_by_group(
            ax,
            df,
            feature,
            feature,
        )

    fig.suptitle(
        "Descriptive Statistics by Clinical Group",
        fontsize=15,
        fontweight="bold",
    )

    output_path = OUTPUT_DIR / "descriptive_statistics.png"

    fig.savefig(output_path, dpi=300)

    plt.close(fig)

    print(f"Saved {output_path.name}")


def plot_pti_scatter(df: pd.DataFrame) -> None:

    fig, ax = plt.subplots(
        figsize=(7, 6),
        constrained_layout=True,
    )

    for group_name in GROUP_ORDER:

        subset = df[df["group"] == group_name]

        ax.scatter(
            subset["whole_global_peak_pressure"],
            subset["whole_total_pti"],
            label=group_name,
            color=GROUP_COLORS[group_name],
            s=60,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.75,
        )

    ax.set_xlabel("Whole Foot Peak Pressure")
    ax.set_ylabel("Whole Foot Total PTI")
    ax.set_title("Total PTI vs Peak Pressure")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Group")

    output_path = OUTPUT_DIR / "pti_scatter_total_vs_peak.png"

    fig.savefig(output_path, dpi=300)

    plt.close(fig)

    print(f"Saved {output_path.name}")


def plot_region_ratio(df: pd.DataFrame) -> None:

    ratio_df = df.copy()

    ratio_df["heel_fore_pti_ratio"] = (
        ratio_df["heel_total_pti"]
        / ratio_df["fore_total_pti"].replace(0, np.nan)
    )

    fig, ax = plt.subplots(
        figsize=(7, 5),
        constrained_layout=True,
    )

    _scatter_by_group(
        ax,
        ratio_df,
        "heel_fore_pti_ratio",
        "Heel / Forefoot PTI Ratio",
    )

    output_path = OUTPUT_DIR / "pti_region_ratio.png"

    fig.savefig(output_path, dpi=300)

    plt.close(fig)

    print(f"Saved {output_path.name}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Plot feature graphs from extracted CSV."
    )

    parser.add_argument(
        "--skip-boxplots",
        action="store_true",
        help="Only save overview/scatter plots.",
    )

    args = parser.parse_args()

    setup_matplotlib()

    ensure_dir(OUTPUT_DIR)

    df = load_features()

    plot_descriptive_overview(df)

    plot_pti_scatter(df)

    plot_region_ratio(df)

    if not args.skip_boxplots:
        plot_feature_boxplots(df)


if __name__ == "__main__":
    main()