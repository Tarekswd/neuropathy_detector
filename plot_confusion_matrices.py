from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIRS = [PROJECT_ROOT / "models" / "tuned", PROJECT_ROOT / "models" / "grouped_tuned"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot confusion matrices for saved model metrics.")
    parser.add_argument(
        "--metrics-dirs",
        type=Path,
        nargs="*",
        default=DEFAULT_METRICS_DIRS,
        help="Directories containing saved metrics JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where confusion matrix plots will be saved.",
    )
    parser.add_argument(
        "--include-grouped",
        action="store_true",
        help="Include grouped tuning metrics from models/grouped_tuned.",
    )
    parser.add_argument(
        "--include-ungrouped",
        action="store_true",
        help="Include ungrouped tuning metrics from models/tuned.",
    )
    return parser.parse_args()


def find_metrics_files(metrics_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for metrics_dir in metrics_dirs:
        if not metrics_dir.exists():
            continue
        files.extend(sorted(metrics_dir.glob("*metrics.json")))
    return sorted(files)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_confusion_matrix(
    ax,
    matrix: np.ndarray,
    labels: list,
    title: str,
) -> None:
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    fmt = "d"
    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(int(matrix[i, j]), fmt),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def get_cm_labels(metrics: dict, data_key: str = "test_metrics") -> list[str]:
    labels = metrics.get(data_key, {}).get("confusion_matrix_labels")
    if labels is None:
        return []
    return [str(label) for label in labels]


def get_confusion_matrix(metrics: dict, data_key: str = "test_metrics") -> np.ndarray:
    cm = metrics.get(data_key, {}).get("confusion_matrix")
    if cm is None:
        raise ValueError(f"No confusion matrix found for {data_key}")
    return np.asarray(cm, dtype=int)


def should_include_file(path: Path, include_grouped: bool, include_ungrouped: bool) -> bool:
    if include_grouped and include_ungrouped:
        return True
    if include_grouped:
        return "grouped" in path.name
    if include_ungrouped:
        return "grouped" not in path.name
    return True


def build_output_filename(metrics: dict, source: Path) -> str:
    model = metrics.get("model", "model")
    task = metrics.get("task", "task")
    validation = metrics.get("validation", "validation")
    source_name = source.stem.replace("_metrics", "")
    return f"confusion_matrix_{model}_{task}_{validation}.png"


def render_plots(metrics_path: Path, output_dir: Path) -> None:
    metrics = load_json(metrics_path)
    model = metrics.get("model", "model")
    task = metrics.get("task", "task")
    validation = metrics.get("validation", "validation")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = build_output_filename(metrics, metrics_path)
    output_path = output_dir / file_name

    test_cm = get_confusion_matrix(metrics, "test_metrics")
    test_labels = get_cm_labels(metrics, "test_metrics")
    if not test_labels:
        test_labels = [str(i) for i in range(test_cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    plot_confusion_matrix(
        ax,
        test_cm,
        test_labels,
        title=f"{model.upper()} {task} test",
    )

    fig.suptitle(f"Test confusion matrix: {model.upper()} ({task}) [{validation}]")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if args.include_grouped or args.include_ungrouped:
        metrics_dirs = []
        if args.include_ungrouped:
            metrics_dirs.append(PROJECT_ROOT / "models" / "tuned")
        if args.include_grouped:
            metrics_dirs.append(PROJECT_ROOT / "models" / "grouped_tuned")
    else:
        metrics_dirs = args.metrics_dirs

    metrics_files = [path for path in find_metrics_files(metrics_dirs) if should_include_file(path, args.include_grouped, args.include_ungrouped)]
    if not metrics_files:
        raise FileNotFoundError(f"No metrics JSON files found in {metrics_dirs}")

    print(f"Found {len(metrics_files)} metrics files. Generating confusion matrices...")
    for metrics_path in metrics_files:
        try:
            render_plots(metrics_path, output_dir)
            print(f"Saved: {output_dir / build_output_filename(load_json(metrics_path), metrics_path)}")
        except Exception as exc:
            print(f"Skipping {metrics_path}: {exc}")

if __name__ == "__main__":
    main()
