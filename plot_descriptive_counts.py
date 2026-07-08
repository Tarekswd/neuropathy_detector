"""Plot descriptive dataset counts from ml_features/npy_features_raw.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import (
    GROUP_COLORS,
    GROUP_ORDER,
    PROJECT_ROOT,
    ensure_dir,
    setup_matplotlib,
)

DEFAULT_FEATURES_CSV = PROJECT_ROOT / "ml_features" / "npy_features_raw.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "graph_plots"


def load_features(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing features CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"group", "subject_id", "event_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df = df.copy()
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    return df


def counts_by_group(series: pd.Series) -> pd.Series:
    return series.reindex(GROUP_ORDER, fill_value=0).astype(int)


def add_bar_labels(ax, values: pd.Series) -> None:
    max_value = max(int(values.max()), 1)
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + max_value * 0.02,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_group_bar(
    values: pd.Series,
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = [GROUP_COLORS[group_name] for group_name in GROUP_ORDER]

    ax.bar(values.index.astype(str), values.values, color=colors, edgecolor="black", linewidth=0.6)
    add_bar_labels(ax, values)
    ax.set_title(title)
    ax.set_xlabel("Clinical Group")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, max(values.max() * 1.15, 1))

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_footprints_per_patient(df: pd.DataFrame, output_path: Path) -> None:
    patient_counts = (
        df.groupby(["group", "subject_id"], observed=False)
        .size()
        .reset_index(name="footprints")
        .sort_values(["group", "subject_id"])
    )
    patient_counts = patient_counts[patient_counts["footprints"] > 0]

    labels = patient_counts["subject_id"].astype(str).tolist()
    values = patient_counts["footprints"].astype(int).tolist()
    colors = [GROUP_COLORS[str(group_name)] for group_name in patient_counts["group"]]

    width = max(10, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(width, 5.5), constrained_layout=True)

    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_title("Number of Footprints per Patient")
    ax.set_xlabel("Patient")
    ax.set_ylabel("Number of Footprints")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", labelrotation=90)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[group_name], ec="black")
        for group_name in GROUP_ORDER
        if group_name in set(patient_counts["group"].astype(str))
    ]
    legend_labels = [
        group_name
        for group_name in GROUP_ORDER
        if group_name in set(patient_counts["group"].astype(str))
    ]
    ax.legend(legend_handles, legend_labels, title="Group", ncols=len(legend_labels), loc="upper right")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_descriptive_counts(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)

    patients_per_group = counts_by_group(df.groupby("group", observed=False)["subject_id"].nunique())
    footprints_per_group = counts_by_group(df.groupby("group", observed=False).size())

    plot_group_bar(
        patients_per_group,
        title="Number of Patients per Group",
        ylabel="Number of Patients",
        output_path=output_dir / "patients_per_group.png",
    )
    plot_group_bar(
        footprints_per_group,
        title="Number of Footprints per Group",
        ylabel="Number of Footprints",
        output_path=output_dir / "footprints_per_group.png",
    )
    plot_footprints_per_patient(df, output_dir / "footprints_per_patient.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create descriptive count graphs for patients and footprints."
    )
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_matplotlib()
    df = load_features(args.features_csv)
    plot_descriptive_counts(df, args.output_dir)


if __name__ == "__main__":
    main()
