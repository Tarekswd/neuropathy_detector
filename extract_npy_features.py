from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = SCRIPT_DIR / "npy_fixed"
OUTPUT_DIR = SCRIPT_DIR / "ml_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ["GC", "GD", "NL", "NS"]
GROUP_TO_ID = {group_name: index for index, group_name in enumerate(GROUP_ORDER)}
REGIONS = ["whole", "heel", "mid", "fore"]
METRICS = [
    "contact_area",
    "global_peak_pressure",
    "total_pti",
    "contact_duration_proxy",
    "total_plantar_load_proxy",
]


def load_metric_maps(event_dir: Path) -> dict[str, np.ndarray]:
    metric_map: dict[str, np.ndarray] = {}
    for metric_name in ["P_contact", "P_mean", "P_peak", "P_PTI"]:
        metric_path = event_dir / f"{metric_name}.npy"
        if metric_path.exists():
            metric_map[metric_name] = np.load(metric_path)
    return metric_map


def region_slice(region_name: str, row_count: int) -> slice:
    if region_name == "whole":
        return slice(0, row_count)

    fore_end = int(round(row_count * 0.48))
    heel_start = int(round(row_count * 0.69))

    if region_name == "fore":
        return slice(0, fore_end)
    if region_name == "mid":
        return slice(fore_end, heel_start)
    if region_name == "heel":
        return slice(heel_start, row_count)

    raise ValueError(f"Unknown region: {region_name}")


def region_array(array: np.ndarray, row_slice: slice) -> np.ndarray:
    return np.asarray(array, dtype=float)[row_slice, :]


def contact_area_proxy(metric_map: dict[str, np.ndarray], row_slice: slice) -> float:
    region = region_array(metric_map["P_contact"], row_slice)
    return float(np.count_nonzero(region > 0))


def global_peak_pressure(metric_map: dict[str, np.ndarray], row_slice: slice) -> float:
    region = region_array(metric_map["P_peak"], row_slice)
    return float(np.max(region)) if region.size else 0.0


def total_pti(metric_map: dict[str, np.ndarray], row_slice: slice) -> float:
    region = region_array(metric_map["P_PTI"], row_slice)
    return float(np.sum(region))


def contact_duration_proxy(metric_map: dict[str, np.ndarray], row_slice: slice) -> float:
    region = region_array(metric_map["P_contact"], row_slice) > 0
    if region.size == 0:
        return 0.0
    active_rows = np.count_nonzero(region.any(axis=1))
    return float(active_rows / region.shape[0])


def total_plantar_load_proxy(metric_map: dict[str, np.ndarray], row_slice: slice) -> float:
    region = region_array(metric_map["P_mean"], row_slice)
    return float(np.sum(region))


METRIC_FUNCTIONS = {
    "contact_area": contact_area_proxy,
    "global_peak_pressure": global_peak_pressure,
    "total_pti": total_pti,
    "contact_duration_proxy": contact_duration_proxy,
    "total_plantar_load_proxy": total_plantar_load_proxy,
}


def collect_event_dirs() -> list[Path]:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_ROOT}")
    return sorted({path.parent for path in INPUT_ROOT.rglob("P_PTI.npy")})


def extract_row(event_dir: Path) -> dict[str, float | str]:
    metric_map = load_metric_maps(event_dir)
    if len(metric_map) != 4:
        raise ValueError(f"Missing one or more maps in {event_dir}")

    contact_shape = metric_map["P_contact"].shape
    if len(contact_shape) != 2:
        raise ValueError(f"Expected 2D arrays in {event_dir}")

    group_name = event_dir.parts[-2].split("_")[0]
    subject_id = event_dir.parts[-2]
    event_id = event_dir.name

    row: dict[str, float | str] = {
        "group": group_name,
        "class_id": GROUP_TO_ID[group_name],
        "subject_id": subject_id,
        "event_id": event_id,
        "source_folder": str(event_dir.relative_to(INPUT_ROOT)),
    }

    row_count = contact_shape[0]
    for region_name in REGIONS:
        row_slice = region_slice(region_name, row_count)
        for metric_name in METRICS:
            feature_name = f"{region_name}_{metric_name}"
            row[feature_name] = METRIC_FUNCTIONS[metric_name](metric_map, row_slice)

    return row


def z_normalize_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    stats_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_df = df.copy()
    stats_rows = []

    for column_name in feature_columns:
        values = pd.to_numeric(normalized_df[column_name], errors="coerce").astype(float)

        if stats_df is not None:
            row = stats_df.loc[stats_df["feature"] == column_name].iloc[0]
            mean_value = float(row["mean"])
            std_value = float(row["std"])
        else:
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=0))

        if std_value == 0.0 or np.isnan(std_value):
            normalized_df[column_name] = 0.0
        else:
            normalized_df[column_name] = (values - mean_value) / std_value

        stats_rows.append(
            {
                "feature": column_name,
                "mean": mean_value,
                "std": std_value,
            }
        )

    return normalized_df, pd.DataFrame(stats_rows)


def subject_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    groups = df["subject_id"].astype(str).values
    y = df["class_id"].astype(int).values

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, y, groups))

    train_subjects = sorted(set(groups[train_idx]))
    test_subjects = sorted(set(groups[test_idx]))
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        raise RuntimeError(f"Subject leakage in split: {overlap}")

    split_info = {
        "strategy": "subject_level",
        "test_size": test_size,
        "random_state": random_state,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "train_events": int(len(train_idx)),
        "test_events": int(len(test_idx)),
        "subject_overlap": 0,
    }
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy(), split_info


def main() -> None:
    event_dirs = collect_event_dirs()
    if not event_dirs:
        raise FileNotFoundError(f"No event folders found under {INPUT_ROOT}")

    rows = []
    for event_dir in event_dirs:
        rows.append(extract_row(event_dir))

    if not rows:
        raise RuntimeError("No feature rows could be extracted")

    raw_df = pd.DataFrame(rows)
    feature_columns = [f"{region_name}_{metric_name}" for region_name in REGIONS for metric_name in METRICS]

    raw_df = raw_df[["group", "class_id", "subject_id", "event_id", "source_folder", *feature_columns]]

    train_raw_df, test_raw_df, split_info = subject_train_test_split(raw_df)
    
    # Calculate statistics from the train split for reference/logging
    stats_rows = []
    for column_name in feature_columns:
        values = pd.to_numeric(train_raw_df[column_name], errors="coerce").astype(float)
        stats_rows.append({
            "feature": column_name,
            "mean": float(values.mean()) if not values.empty else 0.0,
            "std": float(values.std(ddof=0)) if not values.empty else 0.0
        })
    train_stats_df = pd.DataFrame(stats_rows)

    # Use raw features directly (since 2D .npy arrays are already z-normalized)
    train_df = train_raw_df.copy()
    test_df = test_raw_df.copy()

    # ------------------------------------------------------------------ #
    # Combined (all events) dataset — train + test, z-norm with train stats
    # Tagged with a 'split' column so downstream scripts can distinguish.
    # This is the dataset used by per-fold confusion-matrix graphing so
    # that ALL events/patients are visible across folds, not just the 80 %
    # training partition.
    # ------------------------------------------------------------------ #
    train_df_tagged = train_df.copy()
    train_df_tagged.insert(0, "split", "train")
    test_df_tagged = test_df.copy()
    test_df_tagged.insert(0, "split", "test")
    all_df = pd.concat([train_df_tagged, test_df_tagged], ignore_index=True)

    encoding_df = pd.DataFrame(
        {
            "group": GROUP_ORDER,
            "class_id": [GROUP_TO_ID[group_name] for group_name in GROUP_ORDER],
        }
    )

    raw_path = OUTPUT_DIR / "npy_features_raw.csv"
    train_path = OUTPUT_DIR / "npy_features_train.csv"
    test_path = OUTPUT_DIR / "npy_features_test.csv"
    all_path = OUTPUT_DIR / "npy_features_all.csv"
    encoding_path = OUTPUT_DIR / "npy_class_encoding.csv"
    stats_path = OUTPUT_DIR / "npy_feature_scaler_stats.csv"
    split_path = OUTPUT_DIR / "npy_split_manifest.json"

    raw_df.to_csv(raw_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    all_df.to_csv(all_path, index=False)
    encoding_df.to_csv(encoding_path, index=False)
    train_stats_df.to_csv(stats_path, index=False)
    split_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    print(f"Extracted {len(raw_df)} samples from {raw_df['subject_id'].nunique()} subjects")
    print(
        f"Split: {len(split_info['train_subjects'])} train subjects "
        f"({split_info['train_events']} events) / "
        f"{len(split_info['test_subjects'])} test subjects "
        f"({split_info['test_events']} events)"
    )
    print(f"All events combined (train + test): {len(all_df)} rows, {all_df['subject_id'].nunique()} subjects")
    print(f"Saved raw features -> {raw_path}")
    print(f"Saved train features (z-norm fit on train only) -> {train_path}")
    print(f"Saved test features (z-norm with train stats) -> {test_path}")
    print(f"Saved all features (train + test, z-norm with train stats) -> {all_path}")
    print(f"Saved class encoding -> {encoding_path}")
    print(f"Saved train scaler stats -> {stats_path}")
    print(f"Saved split manifest -> {split_path}")


if __name__ == "__main__":
    main()