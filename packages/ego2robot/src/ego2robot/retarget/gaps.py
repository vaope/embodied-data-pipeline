"""Gap handling: short gaps interpolate, long gaps blend to the home pose.

Paper appendix A.1::

    Large gaps (> 10 frames) are filled with the robot's home configuration,
    with smooth blending transitions at the gap boundaries.
    n = max(5, min(90, ceil(0.6 * n_pos + 0.4 * n_rot)))
    n_pos = dp / 3.25 mm,  n_rot = dtheta / 1.08 deg
    Small gaps (<= 10 frames) use linear position interpolation + SLERP.

The paper does not publish the home pose itself, so it comes from configuration
(ADR-0006).  For the ***hand*** side of the pipeline the home pose is a canonical
gripper pose in the camera frame; the robot-side home is the Panda ``home_qpos``
resolved through FK, which is what M3 feeds in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.transforms import exp_so3, log_so3

__all__ = ["GapConfigLike", "GapFillResult", "blend_length_frames", "fill_gaps"]


@dataclass
class GapConfigLike:
    small_gap_max_frames: int = 10
    blend_min_frames: int = 5
    blend_max_frames: int = 90
    blend_position_speed_m_per_frame: float = 0.00325  # 3.25 mm
    blend_rotation_speed_rad_per_frame: float = 0.01885  # 1.08 deg
    blend_position_weight: float = 0.6
    blend_rotation_weight: float = 0.4
    home_gripper_width_m: float = 0.08
    home_source: str = "config"


@dataclass
class GapFillResult:
    """Trajectory with gaps resolved, plus bookkeeping of what was filled."""

    position_C_m: NDArray[np.float64]
    rotation_C_TCP: NDArray[np.float64]
    width_m: NDArray[np.float64]
    filled: NDArray[np.bool_]
    gap_spans: list[tuple[int, int, int]] = field(default_factory=list)
    """``(start, stop_exclusive, length)`` for every gap that was filled."""
    small_gaps: int = 0
    large_gaps: int = 0
    blend_lengths: list[int] = field(default_factory=list)
    home_was_derived: bool = False
    """True when the caller supplied no home pose and it was derived from the
    trajectory centroid (recorded rather than silently assumed)."""


def blend_length_frames(
    delta_position_m: float,
    delta_rotation_rad: float,
    cfg: GapConfigLike | None = None,
) -> int:
    """Paper appendix A.1 blend length, clamped to ``[5, 90]`` frames."""
    cfg = cfg or GapConfigLike()
    n_pos = delta_position_m / cfg.blend_position_speed_m_per_frame if cfg.blend_position_speed_m_per_frame else 0.0
    n_rot = delta_rotation_rad / cfg.blend_rotation_speed_rad_per_frame if cfg.blend_rotation_speed_rad_per_frame else 0.0
    raw = cfg.blend_position_weight * n_pos + cfg.blend_rotation_weight * n_rot
    return int(max(cfg.blend_min_frames, min(cfg.blend_max_frames, int(np.ceil(raw)))))


def _slerp(R_a: NDArray[np.float64], R_b: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Shortest-arc SLERP between two rotation matrices."""
    rel = log_so3(R_a.T @ R_b)
    return R_a @ exp_so3(alpha * rel)


def _gap_spans(valid: NDArray[np.bool_]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for t, ok in enumerate(valid):
        if not ok and start is None:
            start = t
        elif ok and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, len(valid)))
    return spans


def fill_gaps(
    position_C_m: NDArray[np.float64],
    rotation_C_TCP: NDArray[np.float64],
    width_m: NDArray[np.float64],
    valid: NDArray[np.bool_],
    *,
    cfg: GapConfigLike | None = None,
    home_position_C_m: NDArray[np.float64] | None = None,
    home_rotation_C_TCP: NDArray[np.float64] | None = None,
    home_width_m: float | None = None,
) -> GapFillResult:
    """Fill invalid runs.

    * ``length <= small_gap_max_frames``: linear interpolation in position and
      SLERP in orientation between the bracketing valid frames.
    * ``length > small_gap_max_frames``: the run is filled with the home pose and
      the first/last ``n`` frames blend smoothly out of / into the neighbours,
      with ``n`` from :func:`blend_length_frames`.
    * A run that starts at frame 0 or extends to the end has only one bracket and
      is treated as a large gap (there is nothing to interpolate towards).
    """
    cfg = cfg or GapConfigLike()
    position = np.array(position_C_m, dtype=np.float64, copy=True)
    rotation = np.array(rotation_C_TCP, dtype=np.float64, copy=True)
    width = np.array(width_m, dtype=np.float64, copy=True)
    valid = np.asarray(valid, dtype=bool).copy()
    T = position.shape[0]
    filled = ~valid

    res = GapFillResult(
        position_C_m=position,
        rotation_C_TCP=rotation,
        width_m=width,
        filled=filled,
    )
    if T == 0:
        return res

    if home_position_C_m is None or home_rotation_C_TCP is None:
        # [ENGINEERING] derive a home pose from the trajectory itself when the
        # caller supplied none: the centroid of valid frames, keeping the first
        # valid orientation.  Recorded in the result so it is never silent.
        good = valid & np.isfinite(position).all(axis=1)
        if good.any():
            home_position_C_m = np.mean(position[good], axis=0)
            home_rotation_C_TCP = rotation[np.nonzero(good)[0][0]]
        else:
            home_position_C_m = np.zeros(3, dtype=np.float64)
            home_rotation_C_TCP = np.eye(3, dtype=np.float64)
        res.home_was_derived = True
    if home_width_m is None:
        home_width_m = cfg.home_gripper_width_m

    spans = _gap_spans(valid)
    for start, stop in spans:
        length = stop - start
        res.gap_spans.append((start, stop, length))
        left = start - 1
        right = stop
        has_left = left >= 0 and valid[left]
        has_right = right < T and valid[right]

        if length <= cfg.small_gap_max_frames and has_left and has_right:
            res.small_gaps += 1
            for t in range(start, stop):
                alpha = (t - left) / float(right - left)
                position[t] = (1.0 - alpha) * position[left] + alpha * position[right]
                rotation[t] = _slerp(rotation[left], rotation[right], alpha)
                width[t] = (1.0 - alpha) * width[left] + alpha * width[right]
                valid[t] = True
            continue

        res.large_gaps += 1
        dp = 0.0
        dtheta = 0.0
        if has_left:
            dp = float(np.linalg.norm(np.asarray(home_position_C_m) - position[left]))
            dtheta = float(np.linalg.norm(log_so3(rotation[left].T @ np.asarray(home_rotation_C_TCP))))
        elif has_right:
            dp = float(np.linalg.norm(np.asarray(home_position_C_m) - position[right]))
            dtheta = float(np.linalg.norm(log_so3(np.asarray(home_rotation_C_TCP).T @ rotation[right])))
        n_blend = blend_length_frames(dp, dtheta, cfg)
        res.blend_lengths.append(n_blend)

        for t in range(start, stop):
            position[t] = np.asarray(home_position_C_m, dtype=np.float64)
            rotation[t] = np.asarray(home_rotation_C_TCP, dtype=np.float64)
            width[t] = float(home_width_m)
            valid[t] = True

        if has_left:
            for k in range(1, min(n_blend, length) + 1):
                t = start + k - 1
                alpha = k / float(n_blend)
                position[t] = (1.0 - alpha) * position[left] + alpha * np.asarray(home_position_C_m)
                rotation[t] = _slerp(rotation[left], np.asarray(home_rotation_C_TCP), alpha)
                width[t] = (1.0 - alpha) * width[left] + alpha * float(home_width_m)
        if has_right:
            for k in range(1, min(n_blend, length) + 1):
                t = stop - k
                alpha = k / float(n_blend)
                position[t] = (1.0 - alpha) * position[right] + alpha * np.asarray(home_position_C_m)
                rotation[t] = _slerp(rotation[right], np.asarray(home_rotation_C_TCP), alpha)
                width[t] = (1.0 - alpha) * width[right] + alpha * float(home_width_m)

    res.position_C_m = position
    res.rotation_C_TCP = rotation
    res.width_m = width
    return res
