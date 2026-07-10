from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from extract_npy_features import (
    GROUP_ORDER,
    GROUP_TO_ID,
    collect_event_dirs,
    extract_row,
    z_normalize_dataframe,
)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = SCRIPT_DIR / "npy_fixed"
OUTPUT_ROOT = SCRIPT_DIR / "grouped_patient_footprints"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def aggregate_subject_features(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not event_rows:
        raise ValueError("No event rows were provided for aggregation.")

    first_row = event_rows[0]
    feature_columns = [
        key
        for key in first_row
        if key not in {"group", "class_id", "subject_id", "event_id", "source_folder"}
    ]

    aggregated: dict[str, Any] = {
        "group": first_row["group"],
        "class_id": first_row["class_id"],
        "subject_id": first_row["subject_id"],
        "event_count": len(event_rows),
        "event_ids": ";".join(str(row["event_id"]) for row in event_rows),
        "source_folders": ";".join(str(row["source_folder"]) for row in event_rows),
    }

    for feature_name in feature_columns:
        values = [float(row[feature_name]) for row in event_rows]
        aggregated[f"{feature_name}_mean"] = float(np.mean(values))

    return aggregated


def build_grouped_patient_frame() -> tuple[pd.DataFrame, list[str]]:
    event_dirs = collect_event_dirs()
    if not event_dirs:
        raise FileNotFoundError(f"No event folders found under {INPUT_ROOT}")

    event_rows = [extract_row(event_dir) for event_dir in event_dirs]
    raw_df = pd.DataFrame(event_rows)
    base_feature_columns = [
        col for col in raw_df.columns if col not in {"group", "class_id", "subject_id", "event_id", "source_folder"}
    ]

    grouped_rows: list[dict[str, Any]] = []
    for subject_id, subject_rows in raw_df.groupby("subject_id", sort=True):
        grouped_rows.append(aggregate_subject_features(subject_rows.to_dict(orient="records")))

    grouped_df = pd.DataFrame(grouped_rows)
    feature_columns = [f"{feature_name}_mean" for feature_name in base_feature_columns]
    grouped_df = grouped_df[
        ["group", "class_id", "subject_id", "event_count", "event_ids", "source_folders", *feature_columns]
    ]
    return grouped_df, feature_columns


def subject_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = df["subject_id"].astype(str).values
    y = df["class_id"].astype(int).values

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, y, groups))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    train_subjects = sorted(set(train_df["subject_id"].astype(str)))
    test_subjects = sorted(set(test_df["subject_id"].astype(str)))
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        raise RuntimeError(f"Subject leakage detected between train and test splits: {overlap}")

    split_info = {
        "strategy": "subject_level_grouped_patient",
        "test_size": test_size,
        "random_state": random_state,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "train_patients": int(len(train_subjects)),
        "test_patients": int(len(test_subjects)),
        "train_events": int(train_df["event_count"].sum()),
        "test_events": int(test_df["event_count"].sum()),
        "subject_overlap": 0,
    }
    return train_df, test_df, split_info


def main() -> None:
    grouped_df, feature_columns = build_grouped_patient_frame()
    if grouped_df.empty:
        raise RuntimeError("No grouped patient rows were created.")

    train_df, test_df, split_info = subject_train_test_split(grouped_df)

    _, train_stats_df = z_normalize_dataframe(train_df, feature_columns)
    train_ready_df, _ = z_normalize_dataframe(train_df, feature_columns, stats_df=train_stats_df)
    test_ready_df, _ = z_normalize_dataframe(test_df, feature_columns, stats_df=train_stats_df)

    train_ready_df = pd.concat(
        [
            train_df[["group", "class_id", "subject_id", "event_count", "event_ids", "source_folders"]].reset_index(drop=True),
            train_ready_df[feature_columns],
        ],
        axis=1,
    )
    test_ready_df = pd.concat(
        [
            test_df[["group", "class_id", "subject_id", "event_count", "event_ids", "source_folders"]].reset_index(drop=True),
            test_ready_df[feature_columns],
        ],
        axis=1,
    )

    # ------------------------------------------------------------------ #
    # Combined (all patients) dataset — train + test, z-norm with train stats
    # Tagged with a 'split' column so downstream scripts can distinguish.
    # Used by per-fold confusion-matrix graphing so ALL patients are
    # included, not just the 80 % training partition.
    # ------------------------------------------------------------------ #
    train_tagged = train_ready_df.copy()
    train_tagged.insert(0, "split", "train")
    test_tagged = test_ready_df.copy()
    test_tagged.insert(0, "split", "test")
    all_ready_df = pd.concat([train_tagged, test_tagged], ignore_index=True)

    encoding_df = pd.DataFrame(
        {
            "group": GROUP_ORDER,
            "class_id": [GROUP_TO_ID[group_name] for group_name in GROUP_ORDER],
        }
    )

    raw_path = OUTPUT_ROOT / "grouped_patient_features.csv"
    train_path = OUTPUT_ROOT / "grouped_patient_features_train.csv"
    test_path = OUTPUT_ROOT / "grouped_patient_features_test.csv"
    all_path = OUTPUT_ROOT / "grouped_patient_features_all.csv"
    stats_path = OUTPUT_ROOT / "grouped_patient_feature_scaler_stats.csv"
    encoding_path = OUTPUT_ROOT / "grouped_patient_class_encoding.csv"
    split_path = OUTPUT_ROOT / "grouped_patient_split_manifest.json"

    grouped_df.to_csv(raw_path, index=False)
    train_ready_df.to_csv(train_path, index=False)
    test_ready_df.to_csv(test_path, index=False)
    all_ready_df.to_csv(all_path, index=False)
    train_stats_df.to_csv(stats_path, index=False)
    encoding_df.to_csv(encoding_path, index=False)
    split_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    print(f"Grouped {len(grouped_df)} patients into {OUTPUT_ROOT}")
    print(
        f"Split: {len(split_info['train_subjects'])} train subjects "
        f"({split_info['train_events']} events) / {len(split_info['test_subjects'])} test subjects "
        f"({split_info['test_events']} events)"
    )
    print(f"All patients combined (train + test): {len(all_ready_df)} rows, {all_ready_df['subject_id'].nunique()} subjects")
    print(f"Saved grouped patient features -> {raw_path}")
    print(f"Saved train-ready features (z-norm on train only) -> {train_path}")
    print(f"Saved test-ready features (z-norm using train stats) -> {test_path}")
    print(f"Saved all-patients features (train + test, z-norm with train stats) -> {all_path}")
    print(f"Saved scaler stats -> {stats_path}")
    print(f"Saved split manifest -> {split_path}")


if __name__ == "__main__":
    main()
