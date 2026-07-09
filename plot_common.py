from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
GROUP_ORDER = ["GC", "GD", "NL", "NS"]
GROUP_COLORS = {
    "GC": "#1f77b4",
    "GD": "#ff7f0e",
    "NL": "#2ca02c",
    "NS": "#d62728",
}
REGIONS = ["whole", "heel", "mid", "fore"]
METRICS = [
    "contact_area",
    "global_peak_pressure",
    "total_pti",
    "contact_duration_proxy",
    "total_plantar_load_proxy",
]
XML_NS = {"z": "http://www.zebris.de/measurements"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def plot_pressure_heatmap(
    ax,
    array: np.ndarray,
    *,
    title: str,
    cmap: str = "inferno",
    vmax: float | None = None,
) -> None:
    data = np.asarray(array, dtype=float)
    masked = np.ma.masked_where(data <= 0, data)
    vmax_value = float(vmax if vmax is not None else (masked.max() if masked.count() else 1.0))
    ax.set_facecolor("black")
    ax.figure.patch.set_facecolor("black")
    image = ax.imshow(
        masked,
        origin="upper",
        cmap=cmap,
        vmin=0,
        vmax=max(vmax_value, 1e-6),
        aspect="equal",
    )
    ax.set_title(title, color="white")
    ax.set_xlabel("Width", color="white")
    ax.set_ylabel("Length (toe → heel)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pressure")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
