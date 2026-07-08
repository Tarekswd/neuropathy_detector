

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = SCRIPT_DIR / "NPY_maps_oriented_new"
if (INPUT_ROOT / "NPY_maps_oriented_new").is_dir():
	INPUT_ROOT = INPUT_ROOT / "NPY_maps_oriented_new"

OUTPUT_ROOT = SCRIPT_DIR / "npy_fixed"


def load_reference_map(event_dir: Path) -> np.ndarray:
	ref_path = event_dir / "P_PTI.npy"
	if ref_path.exists():
		return np.load(ref_path)

	npy_files = sorted(event_dir.glob("*.npy"))
	if not npy_files:
		raise FileNotFoundError(f"No NPY files found in {event_dir}")
	return np.load(npy_files[0])


def principal_axis_angle_deg(array: np.ndarray) -> float:
	active = np.asarray(array) > 0
	points = np.argwhere(active)
	if points.shape[0] < 3:
		return 0.0

	values = np.asarray(array, dtype=float)[active]
	weights = values if np.isfinite(values).all() and float(values.sum()) > 0 else None

	y = points[:, 0].astype(float)
	x = points[:, 1].astype(float)
	if weights is None:
		x = x - x.mean()
		y = y - y.mean()
		covariance = np.cov(np.vstack([x, y]))
	else:
		x = x - np.average(x, weights=weights)
		y = y - np.average(y, weights=weights)
		covariance = np.cov(np.vstack([x, y]), aweights=weights)

	eigenvalues, eigenvectors = np.linalg.eigh(covariance)
	principal_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
	return math.degrees(math.atan2(principal_vector[0], principal_vector[1]))


def rotate_to_straight(array: np.ndarray, angle_deg: float) -> np.ndarray:
	if abs(angle_deg) < 1e-9:
		return np.asarray(array)

	rotated = rotate(
		np.asarray(array, dtype=float),
		angle_deg,
		reshape=False,
		order=1,
		mode="constant",
		cval=0.0,
		prefilter=False,
	)
	rotated = np.clip(rotated, 0.0, None)

	if np.issubdtype(np.asarray(array).dtype, np.integer):
		rotated = np.rint(rotated)

	return rotated.astype(np.asarray(array).dtype, copy=False)


def detect_forefoot_rows(straightened: np.ndarray) -> slice:
	active = np.asarray(straightened) > 0
	row_width = active.sum(axis=1)
	h = straightened.shape[0]
	window = max(1, h // 4)

	head_width = float(row_width[:window].mean())
	tail_width = float(row_width[-window:].mean())

	if head_width >= tail_width:
		stop = max(1, int(round(h * 0.38)))
		return slice(0, stop)

	start = max(0, h - max(1, int(round(h * 0.38))))
	return slice(start, h)


def classify_side(reference_map: np.ndarray, angle_deg: float) -> str:
	straightened = rotate_to_straight(reference_map, -angle_deg)
	if straightened.ndim != 2:
		return "right"

	forefoot_rows = detect_forefoot_rows(straightened)
	mid = straightened.shape[1] // 2
	left_mass = float(straightened[forefoot_rows, :mid].sum())
	right_mass = float(straightened[forefoot_rows, mid:].sum())

	return "left" if right_mass > left_mass else "right"


def transform_array(array: np.ndarray, angle_deg: float, flip_left: bool) -> np.ndarray:
	transformed = rotate_to_straight(array, -angle_deg)
	if flip_left:
		transformed = np.fliplr(transformed)
	return transformed


def process_event_folder(event_dir: Path) -> tuple[float, str]:
	reference_map = load_reference_map(event_dir)
	angle_deg = principal_axis_angle_deg(reference_map)
	detected_side = classify_side(reference_map, angle_deg)
	flip_left = detected_side == "left"

	relative_dir = event_dir.relative_to(INPUT_ROOT)
	output_dir = OUTPUT_ROOT / relative_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	for npy_path in sorted(event_dir.glob("*.npy")):
		array = np.load(npy_path)
		transformed = transform_array(array, angle_deg, flip_left)
		np.save(output_dir / npy_path.name, transformed)

	return angle_deg, detected_side


def main() -> None:
	if not INPUT_ROOT.exists():
		raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

	event_dirs = sorted({path.parent for path in INPUT_ROOT.rglob("P_PTI.npy")})
	if not event_dirs:
		event_dirs = sorted({path.parent for path in INPUT_ROOT.rglob("*.npy")})

	OUTPUT_ROOT.mkdir(exist_ok=True)

	print(f"Found {len(event_dirs)} event folders in {INPUT_ROOT}")
	print(f"Writing fixed arrays to {OUTPUT_ROOT}")

	left_count = 0
	right_count = 0

	for index, event_dir in enumerate(event_dirs, start=1):
		try:
			angle_deg, detected_side = process_event_folder(event_dir)
			if detected_side == "left":
				left_count += 1
			else:
				right_count += 1
			print(
				f"[{index}/{len(event_dirs)}] {event_dir.name}: "
				f"angle={angle_deg:.2f} deg, side={detected_side}"
			)
		except Exception as exc:
			print(f"[{index}/{len(event_dirs)}] {event_dir.name}: skipped ({exc})")

	print(
		f"Done. Processed {left_count + right_count} event folders "
		f"({left_count} mirrored, {right_count} kept as right)."
	)


if __name__ == "__main__":
	main()
