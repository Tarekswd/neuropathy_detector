"""compare_best_models.py
Compare the best ungrouped (tune_models) vs grouped (tune_grouped_footprints)
models for each task (binary / multiclass), crown the overall winner per task,
and run SHAP analysis for each winner.

Outputs
-------
model_comparison_tables/cross_comparison_report.json
    Full comparison table with all candidates and the two winners.
shap_analysis/cross_binary/
    SHAP plots for the best-overall binary model.
shap_analysis/cross_multiclass/
    SHAP plots for the best-overall multiclass model.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError from shap)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Data paths ──────────────────────────────────────────────────────────────
UNGROUPED_TRAIN  = PROJECT_ROOT / "ml_features"      / "patient_features_train.csv"
UNGROUPED_TEST   = PROJECT_ROOT / "ml_features"      / "patient_features_test.csv"
GROUPED_TRAIN    = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_train.csv"
GROUPED_TEST     = PROJECT_ROOT / "grouped_patient_footprints" / "grouped_patient_features_test.csv"

# ── Model directories ────────────────────────────────────────────────────────
UNGROUPED_TUNED_DIR  = PROJECT_ROOT / "models" / "tuned"
GROUPED_TUNED_DIR    = PROJECT_ROOT / "models" / "grouped_tuned"
UNGROUPED_BEST_DIR   = PROJECT_ROOT / "best model"
GROUPED_BEST_DIR     = PROJECT_ROOT / "best model" / "grouped"

# ── Output dirs ──────────────────────────────────────────────────────────────
COMPARISON_DIR = PROJECT_ROOT / "model_comparison_tables"
SHAP_OUT_ROOT  = PROJECT_ROOT / "shap_analysis"

TASKS = ["binary", "multiclass"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _safe(value: Any, fmt: str = ".4f") -> str:
    if value is None:
        return "N/A"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def collect_candidates(task: str) -> list[dict]:
    """Return all tuned model candidates for *task* from both sources."""
    candidates: list[dict] = []

    # --- Ungrouped models ---
    for metrics_file in sorted(UNGROUPED_TUNED_DIR.glob(f"*_{task}_metrics.json")):
        try:
            data = _load_json(metrics_file)
            model_name = data.get("model", metrics_file.stem.split("_")[0])
            pat = data.get("patient_metrics", {}) or {}
            ev  = data.get("event_metrics",   {}) or {}
            joblib_path = UNGROUPED_TUNED_DIR / f"{model_name}_{task}_tuned.joblib"
            candidates.append({
                "source":                "ungrouped",
                "model":                 model_name,
                "task":                  task,
                "label":                 f"{model_name.upper()} (ungrouped)",
                "joblib_path":           str(joblib_path),
                "train_data":            str(UNGROUPED_TRAIN),
                "test_data":             str(UNGROUPED_TEST),
                "cv_balanced_accuracy":  data.get("cv_balanced_accuracy"),
                "patient_balanced_accuracy": pat.get("balanced_accuracy"),
                "patient_f1_score":      pat.get("f1_score"),
                "patient_auc":           pat.get("auc"),
                "event_balanced_accuracy": ev.get("balanced_accuracy"),
            })
        except Exception as exc:
            print(f"  [warn] skipping {metrics_file.name}: {exc}")

    # --- Grouped models ---
    for metrics_file in sorted(GROUPED_TUNED_DIR.glob(f"*_{task}_grouped_metrics.json")):
        try:
            data = _load_json(metrics_file)
            model_name = data.get("model", metrics_file.stem.split("_")[0])
            # Ensure we get the short model key, not a pipeline repr
            if len(model_name) > 20:
                model_name = metrics_file.stem.split("_")[0]
            pat = data.get("patient_metrics", {}) or {}
            ev  = data.get("event_metrics",   {}) or {}
            joblib_path = GROUPED_TUNED_DIR / f"{model_name}_{task}_grouped_tuned.joblib"
            candidates.append({
                "source":                "grouped",
                "model":                 model_name,
                "task":                  task,
                "label":                 f"{model_name.upper()} (grouped)",
                "joblib_path":           str(joblib_path),
                "train_data":            str(GROUPED_TRAIN),
                "test_data":             str(GROUPED_TEST),
                "cv_balanced_accuracy":  data.get("cv_balanced_accuracy"),
                "patient_balanced_accuracy": pat.get("balanced_accuracy"),
                "patient_f1_score":      pat.get("f1_score"),
                "patient_auc":           pat.get("auc"),
                "event_balanced_accuracy": ev.get("balanced_accuracy"),
            })
        except Exception as exc:
            print(f"  [warn] skipping {metrics_file.name}: {exc}")

    return candidates


def select_best(candidates: list[dict]) -> dict:
    """Pick best candidate by patient_balanced_accuracy, then cv_balanced_accuracy."""
    def _sort_key(c: dict):
        pb  = c.get("patient_balanced_accuracy") or 0.0
        cvb = c.get("cv_balanced_accuracy")      or 0.0
        return (pb, cvb)

    return max(candidates, key=_sort_key)


def print_comparison_table(task: str, candidates: list[dict], winner: dict) -> None:
    print(f"\n{'='*80}")
    print(f"  {task.upper()} — Full comparison table")
    print(f"{'='*80}")
    header = f"{'Model':<30} {'CV-BalAcc':>10} {'Pat-BalAcc':>12} {'Pat-F1':>8} {'Pat-AUC':>10}"
    print(header)
    print("-" * len(header))
    for c in sorted(candidates, key=lambda x: (x.get("patient_balanced_accuracy") or 0), reverse=True):
        marker = " <<" if c["label"] == winner["label"] else ""
        print(
            f"{c['label']:<30}"
            f" {_safe(c['cv_balanced_accuracy']):>10}"
            f" {_safe(c['patient_balanced_accuracy']):>12}"
            f" {_safe(c['patient_f1_score']):>8}"
            f" {_safe(c['patient_auc']):>10}"
            f"{marker}"
        )
    print(f"\n  >> Winner: {winner['label']}")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP helpers  (mirrored from run_shap_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_explanation_data(pipeline, X: pd.DataFrame):
    if hasattr(pipeline, "named_steps") and len(pipeline.named_steps) > 1:
        try:
            transformed = pipeline[:-1].transform(X)
            if isinstance(transformed, pd.DataFrame):
                return transformed.to_numpy(), transformed.columns.tolist()
            return transformed, [f"f{i}" for i in range(transformed.shape[1])]
        except Exception:
            pass
    return X.to_numpy(dtype=float), X.columns.tolist()


def _build_explainer(pipeline, X_train_np: np.ndarray, feature_names: list[str]):
    model = pipeline.named_steps.get("model", pipeline)
    if hasattr(model, "coef_"):
        try:
            return shap.LinearExplainer(model, X_train_np), feature_names
        except Exception:
            pass
    if hasattr(model, "feature_importances_"):
        try:
            return shap.TreeExplainer(model), feature_names
        except Exception:
            pass
    if hasattr(pipeline, "predict_proba"):
        fn = lambda v: pipeline.predict_proba(v)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        fn = lambda v: pipeline.decision_function(v)
    else:
        fn = lambda v: pipeline.predict(v)
    return shap.Explainer(fn, X_train_np), feature_names


def run_shap(winner: dict, output_dir: Path, max_display: int = 15) -> None:
    """Run SHAP analysis for the winning model and save plots."""
    print(f"\n{'─'*60}")
    print(f"  SHAP analysis: {winner['label']}")
    print(f"  Output dir   : {output_dir}")
    print(f"{'─'*60}")

    joblib_path = Path(winner["joblib_path"])
    if not joblib_path.exists():
        print(f"  [error] joblib not found: {joblib_path}")
        return

    artifact  = joblib.load(joblib_path)
    pipeline  = artifact["model"]
    features  = artifact.get("features", [])
    task      = winner["task"]
    label     = winner["label"]

    # Load data
    from models.training_utils import load_train_test_datasets, load_full_dataset  # noqa: PLC0415
    train_path = Path(winner["train_data"])
    test_path  = Path(winner["test_data"])
    try:
        X_train, X_test, _, _, _, _, feature_cols = load_train_test_datasets(train_path, test_path, task)
    except RuntimeError:
        X_train, _, _, feature_cols = load_full_dataset([train_path], task)
        X_test, _, _, _ = load_full_dataset([test_path], task)

    if features:
        feature_cols = [f for f in features if f in X_train.columns]
    X_train = X_train[feature_cols]
    X_test  = X_test[feature_cols]

    X_train_np, feat_names = _prepare_explanation_data(pipeline, X_train)
    X_test_np,  _          = _prepare_explanation_data(pipeline, X_test)

    explainer, feat_names = _build_explainer(pipeline, X_train_np, feat_names)
    shap_values = explainer(X_test_np)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[2] >= 2 else values[:, :, 0]
    if values.ndim != 2:
        values = values.reshape(values.shape[0], -1)
    if values.shape[1] != len(feat_names):
        values = values[:, :len(feat_names)]

    importance_df = (
        pd.DataFrame({"feature": feat_names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    print("Top 10 SHAP features:")
    print(importance_df.head(10).to_string(index=False))

    # ── Beeswarm plot ────────────────────────────────────────────────────────
    title_suffix = f"{label} ({task})"
    shap.summary_plot(values, X_test_np, feature_names=feat_names, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP beeswarm — {title_suffix}", fontsize=10, y=1.01)
    beeswarm_path = output_dir / "shap_beeswarm.png"
    fig.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved beeswarm -> {beeswarm_path}")

    # ── Bar plot ─────────────────────────────────────────────────────────────
    imp_asc = importance_df.sort_values("mean_abs_shap", ascending=True)
    bar_fig, bar_ax = plt.subplots(figsize=(10, max(4, 0.28 * len(feat_names))))
    bar_ax.barh(imp_asc["feature"], imp_asc["mean_abs_shap"], color="#3b82f6")
    bar_ax.set_xlabel("Mean |SHAP value|")
    bar_ax.set_title(f"Feature importance (mean |SHAP|) — {title_suffix}")
    bar_ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    bar_path = output_dir / "shap_bar.png"
    bar_fig.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close(bar_fig)
    print(f"  Saved bar plot  -> {bar_path}")

    # ── JSON summary ─────────────────────────────────────────────────────────
    summary = {
        "winner_label":  label,
        "winner_source": winner["source"],
        "winner_model":  winner["model"],
        "task":          task,
        "top_features":  importance_df.head(10).to_dict(orient="records"),
    }
    (output_dir / "shap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    winners: dict[str, dict] = {}
    all_candidates: dict[str, list[dict]] = {}

    for task in TASKS:
        print(f"\n{'#'*80}")
        print(f"#  Collecting candidates for task: {task.upper()}")
        print(f"{'#'*80}")
        candidates = collect_candidates(task)
        if not candidates:
            print(f"  [warn] no candidates found for {task}, skipping.")
            continue
        all_candidates[task] = candidates
        winner = select_best(candidates)
        winners[task] = winner
        print_comparison_table(task, candidates, winner)

    # ── Save comparison report ───────────────────────────────────────────────
    report = {
        task: {
            "winner": {k: v for k, v in winners[task].items()},
            "all_candidates": all_candidates[task],
        }
        for task in winners
    }
    report_path = COMPARISON_DIR / "cross_comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved comparison report -> {report_path}")

    # ── SHAP for each winner ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Running SHAP analysis for winners")
    print(f"{'='*80}")
    for task, winner in winners.items():
        shap_dir = SHAP_OUT_ROOT / f"cross_{task}"
        try:
            run_shap(winner, shap_dir)
        except Exception as exc:
            print(f"  [error] SHAP failed for {task} winner ({winner['label']}): {exc}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY — Best models across grouped vs ungrouped")
    print(f"{'='*80}")
    for task, winner in winners.items():
        print(
            f"  {task.upper():<12} -> {winner['label']:<35}"
            f"  pat_bal_acc={_safe(winner['patient_balanced_accuracy'])}"
            f"  f1={_safe(winner['patient_f1_score'])}"
            f"  auc={_safe(winner['patient_auc'])}"
        )


if __name__ == "__main__":
    main()
