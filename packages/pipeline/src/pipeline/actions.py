"""Derived Stage 5 action views; original canonical records remain authoritative.

Positions and rotations are compared in each frame's camera coordinates.  This
is a camera-relative *view*, without head-motion compensation or resampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ACTION_PROFILE = "canonical_delta_C_target_width_m.v1"
LEGACY_ACTION_PROFILE = "e2r_legacy_delta_C_source_width.v1"
CAMERA_FRAME_SEMANTICS = "per_frame_camera_relative_view"


def make_action_views(
    *,
    timestamp_s: np.ndarray,
    T_C_TCP: np.ndarray | None = None,
    requested_width_m: np.ndarray | None = None,
    bound_width_m: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build N-1 adjacent actions, using NaN for unavailable measurements.

    ``T_C_TCP`` is [N,4,4], widths are [N].  The last channel of ``action7``
    is the target frame's actual robot binding width.  The compatibility view
    uses the source frame's canonical width; it is never a normalized command.
    Rotation validation precedes SciPy, whose constructor can otherwise repair
    invalid matrices implicitly.
    """
    ts = np.asarray(timestamp_s, dtype=np.float64)
    if ts.ndim != 1 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError("timestamp_s must be finite, 1-D and strictly increasing")
    n, m = len(ts), max(0, len(ts) - 1)
    action = np.full((m, 7), np.nan, dtype=np.float32)
    legacy = np.full_like(action, np.nan)
    rotation_valid = np.zeros(n, dtype=bool)
    if T_C_TCP is not None:
        transform = np.asarray(T_C_TCP, dtype=np.float64)
        if transform.shape != (n, 4, 4):
            raise ValueError(f"T_C_TCP must have shape {(n, 4, 4)}")
        rot = transform[:, :3, :3]
        finite = np.isfinite(rot).all(axis=(1, 2))
        good_idx = np.flatnonzero(finite)
        if good_idx.size:
            r = rot[good_idx]
            rotation_valid[good_idx] = (
                np.max(np.abs(np.swapaxes(r, 1, 2) @ r - np.eye(3)), axis=(1, 2)) <= 1e-5
            ) & (np.abs(np.linalg.det(r) - 1.0) <= 1e-5)
        dp = np.diff(transform[:, :3, 3], axis=0)
        action[:, :3] = dp
        legacy[:, :3] = dp
        indices = np.flatnonzero(rotation_valid[:-1] & rotation_valid[1:])
        if indices.size:
            relative = rot[indices + 1] @ np.swapaxes(rot[indices], 1, 2)
            dr = Rotation.from_matrix(relative).as_rotvec().astype(np.float32)
            action[indices, 3:6] = dr
            legacy[indices, 3:6] = dr
    if requested_width_m is not None:
        width = np.asarray(requested_width_m, dtype=np.float64)
        if width.shape != (n,):
            raise ValueError("requested_width_m must have shape [N]")
        legacy[:, 6] = width[:-1]
    if bound_width_m is not None:
        width = np.asarray(bound_width_m, dtype=np.float64)
        if width.shape != (n,):
            raise ValueError("bound_width_m must have shape [N]")
        action[:, 6] = width[1:]
    return {
        "source_frame_index": np.arange(m, dtype=np.int64),
        "target_frame_index": np.arange(1, n, dtype=np.int64),
        "timestamp_s": ts[:-1].copy(),
        "next_timestamp_s": ts[1:].copy(),
        "dt_s": np.diff(ts),
        "action7": action,
        "legacy_action7": legacy,
        "action_finite": np.isfinite(action).all(axis=1),
        "rotation_valid": rotation_valid,
    }


def build_transition_arrays(
    *,
    timestamp_s: np.ndarray,
    frame_valid: np.ndarray,
    frame_reason_bits: np.ndarray,
    T_C_TCP: np.ndarray | None = None,
    requested_width_m: np.ndarray | None = None,
    bound_width_m: np.ndarray | None = None,
    non_finite_reason_bit: int = 0,
    geometry_frame_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Propagate both endpoint reasons without bridging any removed frame."""
    arrays = make_action_views(
        timestamp_s=timestamp_s,
        T_C_TCP=T_C_TCP,
        requested_width_m=requested_width_m,
        bound_width_m=bound_width_m,
    )
    n = len(timestamp_s)
    valid = np.asarray(frame_valid, dtype=bool)
    reason = np.asarray(frame_reason_bits, dtype=np.uint64)
    if valid.shape != (n,) or reason.shape != (n,):
        raise ValueError("frame_valid and frame_reason_bits must have shape [N]")
    finite = arrays.pop("action_finite")
    arrays.pop("rotation_valid")
    transition_reason = reason[:-1] | reason[1:]
    transition_reason[~finite] |= np.uint64(non_finite_reason_bit)
    arrays["transition_valid"] = valid[:-1] & valid[1:] & finite
    arrays["reason_bits"] = transition_reason
    if geometry_frame_valid is not None:
        geometry = np.asarray(geometry_frame_valid, dtype=bool)
        if geometry.shape != (n,):
            raise ValueError("geometry_frame_valid must have shape [N]")
        arrays["geometry_candidate_valid"] = geometry[:-1] & geometry[1:] & finite
    return arrays
