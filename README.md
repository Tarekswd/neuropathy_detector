# Plantar Pressure Classification for Diabetic Neuropathy Detection

A machine learning pipeline for classifying patients into neuropathy severity groups using plantar pressure measurements from Zebris gait analysis systems. This project was developed as part of an AI & Deep Learning internship.

---

## Overview

This repository contains a complete, end-to-end ML workflow that:

1. **Pre-processes** raw Zebris XML and `.npy` pressure map files
2. **Extracts** biomechanical features across foot regions (whole foot, heel, midfoot, forefoot)
3. **Trains and tunes** multiple classifiers in both binary and multiclass settings, at both event-level and patient-level
4. **Evaluates** models with subject-grouped cross-validation to prevent data leakage
5. **Generates** detailed reports, comparison tables, per-fold confusion matrices, and SHAP analyses
6. **Visualizes** plantar pressure footprints as heatmaps (PNG) and rollover animations (GIF)

### Classification Groups

| ID | Code | Description |
|----|------|-------------|
| 0  | GC   | Control group (healthy) |
| 1  | GD   | Diabetic, no neuropathy |
| 2  | NL   | Neuropathy — low severity |
| 3  | NS   | Neuropathy — severe |

**Binary task:** healthy/diabetic (GC + GD) vs. neuropathy (NL + NS)  
**Multiclass task:** four-way classification across all groups

---

## Repository Structure

```
neuropathy_detector/
│
├── Data Preparation
│   ├── xml_fix.py                             # Repair malformed Zebris XML exports
│   ├── npy_fix.py                             # Normalize and correct raw .npy pressure maps
│   ├── extract_npy_features.py                # Extract per-region biomechanical features from .npy
│   └── extract_grouped_footprint_features.py  # Aggregate event features into one row per patient
│
├── Visualization
│   ├── plot_xml_footprints.py                 # Max-pressure PNGs and rollover GIFs from XML (hot colormap)
│   ├── plotting_npy_pics.py                   # Sample and group-mean heatmaps from .npy (hot colormap)
│   ├── plotting_npy_graph.py                  # Time-series and metric graphs from .npy data
│   ├── plot_descriptive_counts.py             # Descriptive statistics and count plots
│   ├── plot_confusion_matrices.py             # Confusion matrix visualizations
│   ├── plot_features_grid.py                  # Feature distribution grid plots
│   ├── generate_grouped_confusion_matrices.py # Per-group confusion matrix generation
│   ├── plot_common.py                         # Shared plotting utilities and hot colormap default
│   └── run_all_plots.py                       # Orchestrator: run all visualization scripts
│
├── Model Training & Tuning
│   ├── model_builders.py                      # Model factory (LR, SVM, RF, XGBoost, CatBoost)
│   ├── training_utils.py                      # Pipeline builder, subject CV, metrics, sensitivity
│   ├── tune_models.py                         # Hyperparameter tuning — ungrouped event-level features
│   │                                          #   Phase 1: event-level CV (cv_metrics_summary)
│   │                                          #   Phase 2: patient-level CV (patient_cv_metrics_summary)
│   │                                          #   Saves best model per task by patient CV balanced accuracy
│   └── tune_grouped_footprints.py             # Hyperparameter tuning — grouped patient footprint features
│                                              #   Reports CV sensitivity mean ± std per model
│
├── Evaluation
│   ├── evaluate_tuned_models.py               # Evaluate tuned ungrouped models (event + patient metrics)
│   ├── evaluate_tuned_on_grouped.py           # Evaluate ungrouped tuned models on grouped data
│   ├── per_fold_confusion_matrices.py         # Patient-level per-fold CMs for ungrouped models
│   │                                          #   Output: models/per_fold_confusion_matrices/ungrouped/
│   └── grouped_data_models_fold_confusion_matrices.py
│                                              # Patient-level per-fold CMs for grouped models
│                                              #   Output: models/per_fold_confusion_matrices/grouped/
│
├── SHAP Analysis
│   ├── run_shap_analysis.py                   # SHAP beeswarm + bar plots for ungrouped best model
│   └── run_grouped_shap_analysis.py           # SHAP analysis for grouped best model
│
└── Reporting & Analysis
    ├── generate_model_reports.py              # Full per-model performance reports
    ├── generate_model_comparison_tables.py    # 4 comparison tables (ungrouped/grouped × binary/multiclass)
    │                                          #   Columns: Bal.Acc, Accuracy, Sensitivity, Specificity,
    │                                          #            Precision, F1, AUC — all as mean ± std
    ├── generate_hyperparam_reports.py         # Hyperparameter search result summaries
    └── update_report_metrics.py              # Refresh metric values in existing reports
```

---

## Models

Five classifiers are implemented and tuned, each supporting both **binary** and **multiclass** tasks:

| Key | Model | Notes |
|-----|-------|-------|
| `lr` | Logistic Regression | L2 regularization, class-weight balanced |
| `svm` | Support Vector Machine | RBF kernel, OVR multiclass |
| `rf` | Random Forest | 300 estimators, balanced class weights |
| `xgb` | XGBoost | Gradient boosting with log-loss / mlogloss |
| `catboost` | CatBoost | Native handling of imbalanced classes |

All models are wrapped in a `sklearn` `Pipeline` with optional `SelectKBest` feature selection.

---

## Features

Features are extracted across four plantar regions for each gait event:

| Region | Description |
|--------|-------------|
| `whole` | Full plantar surface |
| `heel` | Posterior 31 % of foot |
| `mid` | Midfoot 29 % of foot |
| `fore` | Anterior 48 % of foot |

**Per-region metrics** extracted from `.npy` pressure maps:

- `contact_area` — number of active pressure cells
- `global_peak_pressure` — maximum recorded pressure
- `total_pti` — pressure–time integral (load accumulation)
- `contact_duration_proxy` — estimated contact duration
- `total_plantar_load_proxy` — total force estimate

This yields **20 features** per gait event (4 regions × 5 metrics).

**Grouped patient features** aggregate all events per subject into a single row (mean per feature), used by `tune_grouped_footprints.py`.

---

## Validation Strategy

To avoid subject-level data leakage, all cross-validation uses **`StratifiedGroupKFold`** where `group = subject_id`. This guarantees that all events from the same subject appear in the same fold only.

All CV scripts use **100 % of the data** (train + test combined) so that the 5 validation folds are disjoint and their confusion matrices sum to the full dataset.

### Two-phase training in `tune_models.py`

| Phase | Data | Saved as |
|-------|------|----------|
| Phase 1 | Event-level rows | `cv_metrics_summary` |
| Phase 2 | Patient-level rows (aggregated) | `patient_cv_metrics_summary` |

Both summaries include mean ± std for all metrics and per-fold confusion matrices. The best model per task is selected by **patient-level CV balanced accuracy**.

---

## Outputs

| Path | Contents |
|------|----------|
| `models/tuned/` | Tuned ungrouped model artifacts (`.joblib`) and metrics (`.json`) |
| `models/grouped_tuned/` | Tuned grouped model artifacts and metrics |
| `models/per_fold_confusion_matrices/ungrouped/` | Patient-level per-fold CMs for ungrouped models |
| `models/per_fold_confusion_matrices/grouped/` | Patient-level per-fold CMs for grouped models |
| `best model/` | Best ungrouped model per task selected by patient CV balanced accuracy |
| `best model/grouped/` | Best grouped model per task |
| `model_comparison_tables/` | 4 PNG comparison tables (ungrouped/grouped × binary/multiclass) |
| `shap_analysis/` | SHAP beeswarm, bar, and dependence plots |
| `output_plots/` | Per-subject max-pressure PNGs and rollover GIFs |
| `graph_plots/` | Group-mean and sample footprint heatmaps, descriptive plots |

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn xgboost catboost joblib matplotlib pillow shap
```

### Typical Workflow

```bash
# 1. Fix raw data
python xml_fix.py
python npy_fix.py

# 2. Extract features
python extract_npy_features.py
python extract_grouped_footprint_features.py

# 3. Visualize pressure maps (hot colormap heatmaps + rollover GIFs)
python run_all_plots.py

# 4. Tune and train models
python tune_models.py                          # all models, both tasks
python tune_grouped_footprints.py              # grouped patient footprints

# 5. Evaluate
python evaluate_tuned_models.py
python evaluate_tuned_on_grouped.py

# 6. Per-fold patient-level confusion matrices
python per_fold_confusion_matrices.py          # ungrouped → models/per_fold_confusion_matrices/ungrouped/
python grouped_data_models_fold_confusion_matrices.py  # grouped → models/per_fold_confusion_matrices/grouped/

# 7. SHAP analysis
python run_shap_analysis.py
python run_grouped_shap_analysis.py

# 8. Reports and comparison tables
python generate_model_reports.py
python generate_model_comparison_tables.py     # 4 tables with mean ± std per metric
python generate_hyperparam_reports.py
```

---

## License

This project was developed for academic and research purposes during an internship. Please contact the author before reusing any part of this codebase.
