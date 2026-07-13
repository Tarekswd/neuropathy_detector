"""Generate NPY footprint heatmaps (sample) into graph_plots/."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common import GROUP_ORDER, PROJECT_ROOT, ensure_dir, plot_pressure_heatmap, setup_matplotlib

OUTPUT_DIR = PROJECT_ROOT / "graph_plots"


def collect_event_dirs(npy_root: Path) -> list[Path]:
    return sorted({path.parent for path in npy_root.rglob("P_peak.npy")})


def group_from_event(event_dir: Path) -> str:
    return event_dir.parts[-2].split("_")[0]


def load_peak_map(event_dir: Path) -> np.ndarray:
    return np.load(event_dir / "P_peak.npy")


def plot_single_sample(event_dirs: list[Path], group_name: str, output_path: Path, title: str) -> None:
    group_events = [path for path in event_dirs if group_from_event(path) == group_name]
    if not group_events:
        return

    rng = np.random.default_rng(42)
    chosen = rng.choice(group_events)

    fig, ax = plt.subplots(figsize=(4.5, 7), constrained_layout=True)
    plot_pressure_heatmap(
        ax,
        load_peak_map(chosen),
        title=f"{group_name} {title}\n{chosen.name}",
        cmap="hot",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NPY footprint heatmaps.")
    parser.parse_args()

    setup_matplotlib()
    ensure_dir(OUTPUT_DIR)

    roots = {
        "rotated": PROJECT_ROOT / "npy_right_rotated",
        "fixed": PROJECT_ROOT / "npy_fixed",
        "raw": PROJECT_ROOT / "NPY_maps_oriented_new" / "NPY_maps_oriented_new",
    }

    for label, root_path in roots.items():
        if not root_path.exists():
            print(f"Skipping {label}, {root_path} does not exist.")
            continue

        event_dirs = collect_event_dirs(root_path)
        if not event_dirs:
            print(f"No NPY events found under {root_path}")
            continue

        for group_name in GROUP_ORDER:
            sample_path = OUTPUT_DIR / f"{label}_{group_name}.png"
            plot_single_sample(event_dirs, group_name, sample_path, title=label.capitalize())
            print(f"Saved {sample_path.name}")


if __name__ == "__main__":
    main()
