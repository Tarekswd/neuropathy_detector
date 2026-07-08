# Plantar Pressure Classification for Diabetic Neuropathy Detection

A machine learning pipeline for classifying patients into neuropathy severity groups using plantar pressure measurements from Zebris gait analysis systems. This project was developed as part of an AI & Deep Learning internship.

---

## Overview

This repository contains a complete, end-to-end ML workflow that:

1. **Pre-processes** raw Zebris XML and `.npy` pressure map files
2. **Extracts** biomechanical features across foot regions (whole foot, heel, midfoot, forefoot)
3. **Trains and tunes** multiple classifiers in both binary and multiclass settings
4. **Evaluates** models with subject-grouped cross-validation to prevent data leakage
5. **Generates** detailed reports, comparison tables, confusion matrices, and SHAP analyses

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
for_github/
│
├── Data Preparation
│   ├── xml_fix.py                             # Repair malformed Zebris XML exports
│   ├── npy_fix.py                             # Normalize and correct raw .npy pressure maps
│   ├── extract_npy_features.py                # Extract per-region biomechanical features
│   └── extract_grouped_footprint_features.py  # Feature extraction for grouped patient data
│
├── Visualization
│   ├── plot_xml_footprints.py                 # Render plantar pressure footprints from XML
│   ├── plotting_npy_pics.py                   # Visualize .npy pressure map images
│   ├── plotting_npy_graph.py                  # Time-series and metric graphs from .npy data
│   ├── plot_descriptive_counts.py             # Descriptive statistics and count plots
│   ├── plot_confusion_matrices.py             # Confusion matrix visualizations
│   ├── generate_grouped_confusion_matrices.py # Per-group confusion matrix generation
│   ├── plot_common.py                         # Shared plotting utilities
│   └── run_all_plots.py                       # Orchestrator: run all visualization scripts
│
├── Model Training & Tuning
│   ├── model_builders.py                      # Model factory (LR, SVM, RF, XGBoost, CatBoost)
│   ├── training_utils.py                      # Pipeline builder, cross-validation, metrics
│   ├── tune_models.py                         # GridSearchCV tuning — ungrouped features
│   └── tune_grouped_footprints.py             # GridSearchCV tuning — grouped footprint features
│
├── Evaluation
│   ├── evaluate_tuned_models.py               # Evaluate best tuned models on held-out test set
│   ├── evaluate_tuned_on_grouped.py           # Evaluate tuned models on grouped data splits
│   ├── per_fold_confusion_matrices.py         # Confusion matrices per CV fold
│   └── grouped_data_models_fold_confusion_matrices.py  # Fold-level matrices for grouped models
│
└── Reporting & Analysis
    ├── generate_model_reports.py              # Full per-model performance reports
    ├── generate_model_comparison_tables.py    # Side-by-side model comparison tables
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

All models are wrapped in a `sklearn` `Pipeline` with `SelectKBest` feature selection.

---

## Features

Features are extracted across four plantar regions for each gait event:

| Region | Description |
|--------|-------------|
| `whole` | Full plantar surface |
| `heel` | Posterior 31 % of foot |
| `mid` | Midfoot 29% of foot |
| `fore` | Anterior 48 % of foot |

**Per-region metrics** extracted from `.npy` pressure maps:

- `contact_area` — number of active pressure cells
- `global_peak_pressure` — maximum recorded pressure
- `total_pti` — pressure–time integral (load accumulation)
- `contact_duration_proxy` — estimated contact duration
- `total_plantar_load_proxy` — total force estimate

This yields **20 features** per gait event (4 regions × 5 metrics).

---

## Validation Strategy

To avoid subject-level data leakage, all cross-validation uses **`StratifiedGroupKFold`** where `group = subject_id`. This guarantees that all steps from the same subject appear in the same fold only.

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn xgboost catboost joblib matplotlib
```

### Typical Workflow

```bash
# 1. Fix raw data
python xml_fix.py
python npy_fix.py

# 2. Extract features
python extract_npy_features.py
python extract_grouped_footprint_features.py

# 3. Tune and train models
python tune_models.py --model rf --task binary
python tune_grouped_footprints.py --model xgb --task multiclass

# 4. Evaluate
python evaluate_tuned_models.py
python evaluate_tuned_on_grouped.py

# 5. Generate reports and plots
python run_all_plots.py
python generate_model_reports.py
python generate_model_comparison_tables.py
python generate_hyperparam_reports.py
```

---

## License

This project was developed for academic and research purposes during an internship. Please contact the author before reusing any part of this codebase.
