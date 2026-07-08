"""Generate NPY footprint heatmaps (sample and group-mean) into graph_plots/."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common import GROUP_ORDER, PROJECT_ROOT, ensure_dir, plot_pressure_heatmap, setup_matplotlib

NPY_ROOT = PROJECT_ROOT / "npy_fixed"
OUTPUT_DIR = PROJECT_ROOT / "graph_plots"


def collect_event_dirs() -> list[Path]:
    return sorted({path.parent for path in NPY_ROOT.rglob("P_peak.npy")})


def group_from_event(event_dir: Path) -> str:
    return event_dir.parts[-2].split("_")[0]


def load_peak_map(event_dir: Path) -> np.ndarray:
    return np.load(event_dir / "P_peak.npy")


def plot_group_samples(event_dirs: list[Path], group_name: str, output_path: Path, samples: int = 4) -> None:
    group_events = [path for path in event_dirs if group_from_event(path) == group_name]
    if not group_events:
        return

    rng = np.random.default_rng(42)
    chosen = list(rng.choice(group_events, size=min(samples, len(group_events)), replace=False))
    fig, axes = plt.subplots(2, 2, figsize=(8, 10), constrained_layout=True)
    vmax = max(float(load_peak_map(path).max()) for path in chosen)

    for ax, event_dir in zip(axes.ravel(), chosen):
        plot_pressure_heatmap(
            ax,
            load_peak_map(event_dir),
            title=event_dir.name,
            vmax=vmax,
        )

    for ax in axes.ravel()[len(chosen) :]:
        ax.axis("off")

    fig.suptitle(f"{group_name} sample peak-pressure footprints", fontsize=13)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def pad_footprint(array: np.ndarray, height: int, width: int) -> np.ndarray:
    padded = np.zeros((height, width), dtype=float)
    row_count, col_count = array.shape
    padded[:row_count, :col_count] = array
    return padded


def plot_group_means(event_dirs: list[Path], group_name: str, output_path: Path) -> None:
    group_events = [path for path in event_dirs if group_from_event(path) == group_name]
    if not group_events:
        return

    maps = [load_peak_map(path) for path in group_events]
    max_height = max(array.shape[0] for array in maps)
    max_width = max(array.shape[1] for array in maps)
    stack = np.stack(
        [pad_footprint(array, max_height, max_width) for array in maps],
        axis=0,
    )
    mean_map = np.mean(stack, axis=0)

    fig, ax = plt.subplots(figsize=(4.5, 7), constrained_layout=True)
    plot_pressure_heatmap(ax, mean_map, title=f"{group_name} mean peak pressure (n={len(group_events)})")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NPY footprint heatmaps.")
    parser.add_argument("--samples", type=int, default=4, help="Sample footprints per group.")
    args = parser.parse_args()

    setup_matplotlib()
    ensure_dir(OUTPUT_DIR)
    event_dirs = collect_event_dirs()
    if not event_dirs:
        raise FileNotFoundError(f"No NPY events found under {NPY_ROOT}")

    for group_name in GROUP_ORDER:
        sample_path = OUTPUT_DIR / f"{group_name}_sample_footprints.png"
        mean_path = OUTPUT_DIR / f"{group_name}_mean_footprints.png"
        plot_group_samples(event_dirs, group_name, sample_path, samples=args.samples)
        plot_group_means(event_dirs, group_name, mean_path)
        print(f"Saved {sample_path.name}, {mean_path.name}")


if __name__ == "__main__":
    main()
