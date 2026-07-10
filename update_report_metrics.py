"""Regenerate report section with full test metrics for every model."""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "report.txt"

METRIC_KEYS = [
    "auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1_score",
]


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


VALID_TUNED_MODELS = {"lr", "svm", "rf", "xgb", "catboost"}


def _load_rows(folder: Path, task: str, tuned: bool) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    pattern = f"*_{task}_metrics.json" if tuned else "*_metrics.json"
    for path in sorted(folder.glob(pattern)):
        if tuned:
            name = path.name.replace(f"_{task}_metrics.json", "")
            if name not in VALID_TUNED_MODELS:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            metrics = data["test_metrics"]
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("task") != task:
                continue
            name = path.stem.replace("_metrics", "")
            metrics = data.get("test_metrics", data)
        rows.append((name, metrics))
    rows.sort(key=lambda item: item[1].get("balanced_accuracy", 0), reverse=True)
    return rows


def _format_block(title: str, rows: list[tuple[str, dict]]) -> list[str]:
    lines = [title, "-" * 80]
    for name, m in rows:
        lines.append(f"\n{name.upper()}")
        lines.append(f"  AUC:               {_fmt(m.get('auc'))}")
        lines.append(f"  Balanced Accuracy: {_fmt(m.get('balanced_accuracy'))}")
        lines.append(f"  Sensitivity:       {_fmt(m.get('sensitivity'))}")
        lines.append(f"  Specificity:       {_fmt(m.get('specificity'))}")
        lines.append(f"  Precision:         {_fmt(m.get('precision'))}")
        lines.append(f"  F1-score:          {_fmt(m.get('f1_score'))}")
        lines.append("  Confusion Matrix:")
        for row in m.get("confusion_matrix", []):
            lines.append(f"    {row}")
        lines.append("  Binary Confusion Matrix (non-neuropathy vs neuropathy):")
        for row in m.get("binary_confusion_matrix", []):
            lines.append(f"    {row}")
    lines.append("")
    return lines


def main() -> None:
    lines = [
        "",
        "11. MODEL METRICS ON 20% HOLDOUT TEST SET",
        "=" * 80,
        "Metrics: AUC, Balanced Accuracy, Sensitivity, Specificity, Precision, F1-score, Confusion Matrix, Binary Confusion Matrix",
        "Validation: fixed 80/20 subject-level split, z-normalization fit on train only.",
        "",
    ]

    for task in ["binary", "multiclass"]:
        lines.append(f"{task.upper()} - DEFAULT MODELS (models/{task}models/)")
        lines.append("-" * 80)
        folder = PROJECT_ROOT / "models" / f"{task}models"
        lines.extend(_format_block("", _load_rows(folder, task, tuned=False)))

        lines.append(f"{task.upper()} - TUNED MODELS (models/tuned/)")
        lines.append("-" * 80)
        tuned_folder = PROJECT_ROOT / "models" / "tuned"
        lines.extend(_format_block("", _load_rows(tuned_folder, task, tuned=True)))

    section = "\n".join(lines)
    text = REPORT.read_text(encoding="utf-8")
    marker = "================================================================================\nEnd of report"
    if "11. MODEL METRICS ON 20% HOLDOUT TEST SET" in text:
        text = re.sub(
            r"\n11\. MODEL METRICS ON 20% HOLDOUT TEST SET.*?(?=\n================================================================================\nEnd of report)",
            "\n" + section,
            text,
            flags=re.S,
        )
    elif "11. TUNED MODEL RESULTS" in text:
        text = re.sub(
            r"\n11\. TUNED MODEL RESULTS.*?(?=\n================================================================================\nEnd of report)",
            "\n" + section,
            text,
            flags=re.S,
        )
    else:
        text = text.replace(marker, section + "\n\n" + marker)
    REPORT.write_text(text, encoding="utf-8")
    print(f"Updated {REPORT}")


if __name__ == "__main__":
    main()
