"""Deterministic chunking, merging, and temporal refinement for binary masks.

The parameter defaults follow the M5 contract in ``system-design.md``. Short
interior gaps are filled by signed-distance interpolation as recorded in
ADR-0011; empty leading, trailing, or long gaps remain empty.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["chunk_windows", "merge_chunk_masks", "refine_masks"]


def _as_int(name: str, value: Any, *, minimum: int) -> int:
    """Validate an integer argument while rejecting bool (an int subclass)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def chunk_windows(
    num_frames: int,
    chunk_frames: int = 400,
    overlap: int = 50,
) -> list[tuple[int, int]]:
    """Return end-exclusive windows covering ``[0, num_frames)`` exactly.

    Adjacent windows overlap by ``overlap`` frames. A final partial window is
    emitted only when it adds uncovered frames, so there is no redundant tail
    window. Zero frames produce an empty list.
    """
    frame_count = _as_int("num_frames", num_frames, minimum=0)
    chunk_size = _as_int("chunk_frames", chunk_frames, minimum=1)
    overlap_size = _as_int("overlap", overlap, minimum=0)
    if overlap_size >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_frames")
    if frame_count == 0:
        return []

    windows: list[tuple[int, int]] = []
    start = 0
    while start < frame_count:
        end = min(start + chunk_size, frame_count)
        windows.append((start, end))
        if end == frame_count:
            break
        start = end - overlap_size
    return windows


def merge_chunk_masks(
    chunks: Sequence[tuple[int, NDArray[np.bool_]]],
    num_frames: int,
) -> NDArray[np.bool_]:
    """Merge ``(start_frame, masks)`` chunks with boolean OR on overlaps.

    Each mask array has shape ``[chunk_frames, height, width]`` and bool dtype.
    Every frame in the requested output must be covered by at least one chunk;
    offsets, frame counts, dtypes, and spatial dimensions are checked.
    """
    frame_count = _as_int("num_frames", num_frames, minimum=1)
    if not isinstance(chunks, Sequence):
        raise TypeError("chunks must be a sequence of (start_frame, masks) pairs")
    if not chunks:
        raise ValueError("chunks must cover every requested frame")

    output: NDArray[np.bool_] | None = None
    covered = np.zeros(frame_count, dtype=bool)
    expected_hw: tuple[int, int] | None = None

    for chunk_index, item in enumerate(chunks):
        if not isinstance(item, Sequence) or len(item) != 2:
            raise TypeError(f"chunks[{chunk_index}] must be a (start_frame, masks) pair")
        start = _as_int(f"chunks[{chunk_index}].start_frame", item[0], minimum=0)
        masks = item[1]
        if not isinstance(masks, np.ndarray):
            raise TypeError(f"chunks[{chunk_index}].masks must be a numpy array")
        if masks.dtype != np.bool_:
            raise TypeError(f"chunks[{chunk_index}].masks must have bool dtype")
        if masks.ndim != 3:
            raise ValueError(f"chunks[{chunk_index}].masks must have shape [T, H, W]")
        chunk_length, height, width = (int(v) for v in masks.shape)
        if chunk_length == 0:
            raise ValueError(f"chunks[{chunk_index}].masks must contain at least one frame")
        if height == 0 or width == 0:
            raise ValueError(f"chunks[{chunk_index}].masks must have non-empty spatial dimensions")
        end = start + chunk_length
        if end > frame_count:
            raise ValueError(
                f"chunks[{chunk_index}] covers [{start}, {end}), beyond num_frames={frame_count}"
            )
        hw = (height, width)
        if expected_hw is None:
            expected_hw = hw
            output = np.zeros((frame_count, height, width), dtype=bool)
        elif hw != expected_hw:
            raise ValueError(
                f"chunks[{chunk_index}] has spatial shape {hw}; expected {expected_hw}"
            )

        assert output is not None
        output[start:end] |= masks
        covered[start:end] = True

    if not bool(np.all(covered)):
        missing = np.flatnonzero(~covered)
        preview = ", ".join(str(int(i)) for i in missing[:10])
        suffix = "..." if missing.size > 10 else ""
        raise ValueError(f"chunks leave frame coverage gaps at indices {preview}{suffix}")
    assert output is not None
    return output


def _signed_distance(mask: NDArray[np.bool_], cv2: Any) -> NDArray[np.float32]:
    """Compute inside-minus-outside Euclidean distance, with an exterior pad."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False).astype(np.uint8)
    distance_args = (cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    inside = cv2.distanceTransform(padded, *distance_args)
    outside = cv2.distanceTransform(1 - padded, *distance_args)
    signed = inside - outside
    return np.asarray(signed[1:-1, 1:-1], dtype=np.float32)


def _local_median(values: NDArray[np.int64], window: int) -> NDArray[np.float64]:
    """Compute a clipped, centered temporal median for each frame."""
    half = window // 2
    medians = np.empty(values.shape[0], dtype=np.float64)
    for frame in range(values.shape[0]):
        start = max(0, frame - half)
        end = min(values.shape[0], frame + half + 1)
        medians[frame] = float(np.median(values[start:end]))
    return medians


def refine_masks(
    raw: NDArray[np.bool_],
    *,
    gap_fill_max_frames: int = 3,
    area_median_window: int = 11,
    area_median_fraction: float = 0.5,
    close_kernel: int = 5,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Refine a ``[T, H, W]`` bool mask sequence and return JSON-safe metrics.

    Processing order:

    1. Fill only interior all-empty runs no longer than ``gap_fill_max_frames``
       by linearly interpolating the signed distance fields of the two flank
       masks and thresholding at zero.
    2. For non-empty masks below ``area_median_fraction`` times their clipped,
       centered local area median, copy the nearest frozen candidate mask. A
       candidate must be non-empty and meet its own local threshold; equal
       distances choose the earlier frame. Empty masks are never area-filled.
    3. Apply a per-frame morphological close with an elliptical kernel.

    ``cv2`` is imported only when a non-empty sequence needs processing. The
    input is never modified. Metrics contain ordinary Python scalars/lists so
    they can be serialized as JSON without a custom encoder.
    """
    if not isinstance(raw, np.ndarray):
        raise TypeError("raw must be a numpy array")
    if raw.dtype != np.bool_:
        raise TypeError("raw must have bool dtype")
    if raw.ndim != 3:
        raise ValueError("raw must have shape [T, H, W]")
    frame_count, height, width = (int(v) for v in raw.shape)
    if height == 0 or width == 0:
        raise ValueError("raw must have non-empty spatial dimensions")

    gap_limit = _as_int("gap_fill_max_frames", gap_fill_max_frames, minimum=0)
    median_window = _as_int("area_median_window", area_median_window, minimum=1)
    close_size = _as_int("close_kernel", close_kernel, minimum=1)
    if median_window % 2 == 0:
        raise ValueError("area_median_window must be odd for a centered window")
    if close_size % 2 == 0:
        raise ValueError("close_kernel must be odd")
    if isinstance(area_median_fraction, (bool, np.bool_)) or not isinstance(
        area_median_fraction, Real
    ):
        raise TypeError("area_median_fraction must be a real number")
    fraction = float(area_median_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("area_median_fraction must be finite and within [0, 1]")

    result = raw.copy()
    if frame_count == 0:
        empty: list[int] = []
        return result, {
            "num_frames": 0,
            "height": height,
            "width": width,
            "raw_area_pixels": empty,
            "area_threshold_pixels": [],
            "area_replaced_frame_indices": empty,
            "gap_interpolated_frame_indices": empty,
            "gap_filled_frame_indices": empty,
            "retained_empty_frame_indices": empty,
            "area_before_close_pixels": empty,
            "refined_area_pixels": empty,
        }

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("mask refinement requires OpenCV (opencv-python-headless)") from exc

    raw_areas = np.count_nonzero(raw, axis=(1, 2)).astype(np.int64)
    interpolated_frames: list[int] = []
    filled_frames: list[int] = []

    # Find empty runs in the original sequence so fills cannot chain outward.
    if gap_limit:
        frame = 0
        while frame < frame_count:
            if raw_areas[frame] != 0:
                frame += 1
                continue
            start = frame
            while frame < frame_count and raw_areas[frame] == 0:
                frame += 1
            end = frame
            gap_length = end - start
            if gap_length > gap_limit or start == 0 or end == frame_count:
                continue

            left_distance = _signed_distance(raw[start - 1], cv2)
            right_distance = _signed_distance(raw[end], cv2)
            for target in range(start, end):
                alpha = (target - start + 1) / (gap_length + 1)
                signed = (1.0 - alpha) * left_distance + alpha * right_distance
                result[target] = signed > 0.0
                interpolated_frames.append(target)
                if bool(np.any(result[target])):
                    filled_frames.append(target)

    # Freeze masks and donor eligibility before replacing any small-area mask.
    candidate_masks = result.copy()
    candidate_areas = np.count_nonzero(candidate_masks, axis=(1, 2)).astype(np.int64)
    thresholds = fraction * _local_median(candidate_areas, median_window)
    donor_frames = np.flatnonzero(
        (candidate_areas > 0) & (candidate_areas.astype(np.float64) >= thresholds)
    )
    replaced_frames: list[int] = []
    for frame in range(frame_count):
        area = int(candidate_areas[frame])
        if area == 0 or float(area) >= thresholds[frame] or donor_frames.size == 0:
            continue
        # np.flatnonzero is sorted; argmin returns the first tie, hence earlier.
        nearest_index = int(np.argmin(np.abs(donor_frames - frame)))
        donor = int(donor_frames[nearest_index])
        result[frame] = candidate_masks[donor]
        replaced_frames.append(frame)

    area_before_close = np.count_nonzero(result, axis=(1, 2)).astype(np.int64)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        for frame in range(frame_count):
            result[frame] = cv2.morphologyEx(
                result[frame].astype(np.uint8), cv2.MORPH_CLOSE, kernel
            ).astype(bool)

    refined_areas = np.count_nonzero(result, axis=(1, 2)).astype(np.int64)
    retained_empty = np.flatnonzero(refined_areas == 0)
    metrics: dict[str, Any] = {
        "num_frames": frame_count,
        "height": height,
        "width": width,
        "raw_area_pixels": [int(v) for v in raw_areas],
        "area_threshold_pixels": [float(v) for v in thresholds],
        "gap_interpolated_frame_indices": interpolated_frames,
        "gap_filled_frame_indices": filled_frames,
        "area_replaced_frame_indices": replaced_frames,
        "retained_empty_frame_indices": [int(v) for v in retained_empty],
        "area_before_close_pixels": [int(v) for v in area_before_close],
        "refined_area_pixels": [int(v) for v in refined_areas],
    }
    return result, metrics
