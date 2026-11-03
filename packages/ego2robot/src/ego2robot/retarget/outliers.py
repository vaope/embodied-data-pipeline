"""Velocity-based outlier rejection (paper appendix A.3, equation 7).

::

    tau_t = max(5 * median({v_s}), 0.9 / fps)          (position, m/frame)
    tau_t = max(5 * median({w_s}), 10.0 / fps)         (rotation, rad/frame)

Frames above the threshold are replaced by interpolation from their neighbours
and the whole procedure is iterated **exactly twice**, as the paper states.

The rotated-domain velocity is measured as the geodesic angle between
consecutive orientations, which is the SO(3) analogue of ``|p_t - p_{t-1}|``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.transforms import geodesic_distance_rad, log_so3, exp_so3

__all__ = ["OutlierConfigLike", "OutlierResult", "reject_outliers"]


@dataclass
class OutlierConfigLike:
    enabled: bool = True
    position_median_multiplier: float = 5.0
    rotation_median_multiplier: float = 5.0
    position_floor_per_fps_m: float = 0.9
    rotation_floor_per_fps_rad: float = 10.0
    rounds: int = 2
    min_samples_for_median: int = 3


@dataclass
class OutlierResult:
    """Filtered trajectory plus the mask of frames that were rejected."""

    position_C_m: NDArray[np.float64]
    rotation_C_TCP: NDArray[np.float64]
    width_m: NDArray[np.float64]
    rejected: NDArray[np.bool_]
    position_threshold_m: float
    rotation_threshold_rad: float
    rounds_applied: int
    per_round_rejected: list[int] = field(default_factory=list)

    @property
    def rejected_count(self) -> int:
        return int(np.count_nonzero(self.rejected))


def _interpolate_position(
    position: NDArray[np.float64], bad: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Linear interpolation across ``bad`` frames, using only good samples."""
    out = np.array(position, dtype=np.float64, copy=True)
    good = np.isfinite(out).all(axis=1) & ~bad
    idx = np.nonzero(good)[0]
    if idx.size == 0:
        return out
    if idx.size == 1:
        out[bad] = out[idx[0]]
        return out
    T = out.shape[0]
    for axis in range(3):
        out[:, axis] = np.interp(np.arange(T), idx, out[idx, axis])
    return out


def _interpolate_rotation(
    rotation: NDArray[np.float64], bad: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """SLERP across ``bad`` frames by interpolating rotation vectors from a
    local reference frame, which stays stable for the small gaps a velocity
    filter creates."""
    out = np.array(rotation, dtype=np.float64, copy=True)
    good = np.isfinite(out).all(axis=(1, 2)) & ~bad
    idx = np.nonzero(good)[0]
    if idx.size == 0:
        return out
    if idx.size == 1:
        out[bad] = out[idx[0]]
        return out

    ref = out[idx[0]]
    rotvecs = np.stack([log_so3(ref.T @ out[i]) for i in idx], axis=0)
    T = out.shape[0]
    interp = np.empty((T, 3), dtype=np.float64)
    for axis in range(3):
        interp[:, axis] = np.interp(np.arange(T), idx, rotvecs[:, axis])
    for t in np.nonzero(bad)[0]:
        out[t] = ref @ exp_so3(interp[t])
    return out


def _interpolate_width(width: NDArray[np.float64], bad: NDArray[np.bool_]) -> NDArray[np.float64]:
    good = np.isfinite(width) & ~bad
    idx = np.nonzero(good)[0]
    out = np.array(width, dtype=np.float64, copy=True)
    if idx.size == 0:
        return out
    if idx.size == 1:
        out[bad] = out[idx[0]]
        return out
    out = np.interp(np.arange(width.shape[0]), idx, width[idx])
    return out


def reject_outliers(
    position_C_m: NDArray[np.float64],
    rotation_C_TCP: NDArray[np.float64],
    width_m: NDArray[np.float64],
    valid: NDArray[np.bool_],
    *,
    fps: float,
    cfg: OutlierConfigLike | None = None,
) -> OutlierResult:
    """Run the paper's two-round velocity filter.

    ``position_C_m``/``rotation_C_TCP`` are ``[T, 3]`` / ``[T, 3, 3]``.  Invalid
    frames are excluded from the median so a detection dropout cannot drag the
    threshold up.
    """
    cfg = cfg or OutlierConfigLike()
    position = np.array(position_C_m, dtype=np.float64, copy=True)
    rotation = np.array(rotation_C_TCP, dtype=np.float64, copy=True)
    width = np.array(width_m, dtype=np.float64, copy=True)
    valid = np.asarray(valid, dtype=bool).copy()

    usable = valid & np.isfinite(position).all(axis=1) & np.isfinite(rotation).all(axis=(1, 2))
    T = position.shape[0]
    rejected_total = np.zeros(T, dtype=bool)

    if not cfg.enabled or T < 2 or np.count_nonzero(usable) < cfg.min_samples_for_median + 1:
        return OutlierResult(
            position_C_m=position,
            rotation_C_TCP=rotation,
            width_m=width,
            rejected=rejected_total,
            position_threshold_m=0.0,
            rotation_threshold_rad=0.0,
            rounds_applied=0,
        )

    # --- thresholds from the initial (pre-filter) trajectory ---------------
    steps = np.nonzero(usable[1:] & usable[:-1])[0]
    pos_v = np.linalg.norm(np.diff(position, axis=0), axis=1)[steps] if steps.size else np.zeros(0)
    rot_v = (
        np.array([geodesic_distance_rad(rotation[i], rotation[i + 1]) for i in steps])
        if steps.size
        else np.zeros(0)
    )
    pos_thr = max(
        cfg.position_median_multiplier * float(np.median(pos_v)) if pos_v.size else 0.0,
        cfg.position_floor_per_fps_m / max(fps, 1e-9),
    )
    rot_thr = max(
        cfg.rotation_median_multiplier * float(np.median(rot_v)) if rot_v.size else 0.0,
        cfg.rotation_floor_per_fps_rad / max(fps, 1e-9),
    )

    per_round: list[int] = []
    for _round in range(cfg.rounds):
        bad = np.zeros(T, dtype=bool)
        for t in np.nonzero(usable)[0]:
            if t + 1 < T and usable[t + 1]:
                if float(np.linalg.norm(position[t + 1] - position[t])) > pos_thr:
                    bad[t + 1] = True
                    continue
                if geodesic_distance_rad(rotation[t], rotation[t + 1]) > rot_thr:
                    bad[t + 1] = True
        per_round.append(int(np.count_nonzero(bad)))
        if not bad.any():
            break
        position = _interpolate_position(position, bad)
        rotation = _interpolate_rotation(rotation, bad)
        width = _interpolate_width(width, bad)
        rejected_total |= bad
        usable = usable & ~bad

    return OutlierResult(
        position_C_m=position,
        rotation_C_TCP=rotation,
        width_m=width,
        rejected=rejected_total,
        position_threshold_m=pos_thr,
        rotation_threshold_rad=rot_thr,
        rounds_applied=cfg.rounds,
        per_round_rejected=per_round,
    )
