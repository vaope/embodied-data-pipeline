"""Temporal smoothing: Savitzky-Golay in position/width, Gaussian SLERP in rotation.

Paper appendix A.3::

    Position and gripper width: Savitzky-Golay, window min(21, n),
                                polynomial order min(3, window - 1)
    Orientation: Gaussian-weighted SLERP, sigma = 10 frames, kernel size 21
    Adjacent quaternions are hemisphere-corrected (q_t -> -q_t when q_t.q_{t-1} < 0)

Two engineering details the paper leaves implicit and this module handles
explicitly (implementation-plan.md M2 test matrix):

* the Savitzky-Golay window must be a legal odd number ``<= n`` and greater than
  the polynomial order, so short clips and even windows are repaired rather than
  raising;
* the Gaussian SLERP must never average quaternions with opposite signs, or a
  sign flip becomes a fake 180 degree jump.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from ..core.transforms import (
    exp_so3,
    hemisphere_correct,
    log_so3,
    matrix_from_quat_xyzw,
    quat_xyzw_from_matrix,
)

__all__ = [
    "SmoothingConfigLike",
    "SmoothingResult",
    "gaussian_kernel",
    "legalize_savgol_window",
    "smooth_trajectory",
]


@dataclass
class SmoothingConfigLike:
    position_window: int = 21
    position_polyorder: int = 3
    width_window: int = 21
    width_polyorder: int = 3
    orientation_sigma_frames: float = 10.0
    orientation_kernel_size: int = 21
    enforce_odd_window: bool = True


@dataclass
class SmoothingResult:
    position_C_m: NDArray[np.float64]
    rotation_C_TCP: NDArray[np.float64]
    width_m: NDArray[np.float64]
    applied: dict[str, object] = field(default_factory=dict)
    """What window/polyorder/kernel were actually used, after legalisation."""


def legalize_savgol_window(n: int, window: int, polyorder: int) -> tuple[int, int]:
    """Return a legal ``(window, polyorder)`` for ``n`` samples.

    Rules applied, in order:

    1. ``window <= n``;
    2. ``window`` odd;
    3. ``window > polyorder`` (Savitzky-Golay requires strictly more points than
       the polynomial order);
    4. ``polyorder >= 0``.

    When ``n`` is too small for any smoothing the window collapses to 1, which
    :func:`smooth_trajectory` treats as "leave the signal alone".
    """
    if n <= 0:
        return 1, 0
    window = int(min(window, n))
    if window % 2 == 0:
        window -= 1
    max_poly = max(0, window - 1)
    polyorder = int(min(polyorder, max_poly))
    if window <= polyorder:
        window = polyorder + 1
        if window % 2 == 0:
            window += 1
    if window > n:
        window = n if n % 2 == 1 else max(1, n - 1)
        polyorder = min(polyorder, max(0, window - 1))
    if n < 3:
        return 1, 0
    return window, polyorder


def gaussian_kernel(size: int, sigma: float) -> NDArray[np.float64]:
    """Normalised 1-D Gaussian weights, centre-aligned."""
    if size <= 0:
        raise ValueError("kernel size must be positive")
    if size % 2 == 0:
        size += 1
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    if sigma <= 0:
        w = np.zeros_like(x)
        w[half] = 1.0
        return w
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return w / float(np.sum(w))


def _savgol(
    signal: NDArray[np.float64], window: int, polyorder: int
) -> NDArray[np.float64]:
    if window < 3 or window <= polyorder:
        return np.array(signal, dtype=np.float64, copy=True)
    return savgol_filter(np.asarray(signal, dtype=np.float64), window, polyorder, axis=0, mode="interp")


def _gaussian_slerp_sequence(
    rotation: NDArray[np.float64], *, sigma: float, kernel_size: int
) -> NDArray[np.float64]:
    """Gaussian-weighted SLERP over a rotation sequence.

    Implemented as a weighted average in the tangent space of the *local*
    rotation vector: each rotation is expressed relative to the sequence's
    reference orientation, weighted, renormalised, then mapped back.  Tangent
    averaging avoids the classic pitfall of averaging quaternion components
    (which biases the mean angle) while still allowing the kernel to reach across
    the whole window.
    """
    T = rotation.shape[0]
    if T == 0:
        return rotation
    size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    size = min(size, T if T % 2 == 1 else max(1, T - 1))
    if size < 3:
        return np.array(rotation, dtype=np.float64, copy=True)

    quats = hemisphere_correct(np.stack([quat_xyzw_from_matrix(R) for R in rotation], axis=0))
    ref_q = quats[T // 2]
    ref_R = matrix_from_quat_xyzw(ref_q)
    rotvecs = np.stack([log_so3(ref_R.T @ matrix_from_quat_xyzw(q)) for q in quats], axis=0)

    weights = gaussian_kernel(size, sigma)
    half = size // 2
    smoothed = np.empty_like(rotvecs)
    for t in range(T):
        acc = np.zeros(3, dtype=np.float64)
        wsum = 0.0
        for k in range(-half, half + 1):
            idx = t + k
            if idx < 0 or idx >= T:
                continue
            w = weights[k + half]
            acc += w * rotvecs[idx]
            wsum += w
        smoothed[t] = acc / wsum if wsum > 0 else rotvecs[t]

    return np.stack([ref_R @ exp_so3(rv) for rv in smoothed], axis=0)


def smooth_trajectory(
    position_C_m: NDArray[np.float64],
    rotation_C_TCP: NDArray[np.float64],
    width_m: NDArray[np.float64],
    *,
    valid: NDArray[np.bool_] | None = None,
    cfg: SmoothingConfigLike | None = None,
) -> SmoothingResult:
    """Apply the paper's smoothing to a gap-free trajectory."""
    cfg = cfg or SmoothingConfigLike()
    position = np.asarray(position_C_m, dtype=np.float64)
    rotation = np.asarray(rotation_C_TCP, dtype=np.float64)
    width = np.asarray(width_m, dtype=np.float64)
    T = position.shape[0]
    applied: dict[str, object] = {"num_frames": T}

    if T == 0:
        return SmoothingResult(position, rotation, width, applied)

    finite = np.isfinite(position).all(axis=1) & np.isfinite(width)
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool)
    n_finite = int(np.count_nonzero(finite))
    if n_finite < 3:
        applied["skipped"] = "fewer than 3 finite samples"
        return SmoothingResult(position, rotation, width, applied)

    pos_window, pos_poly = legalize_savgol_window(
        n_finite, cfg.position_window, cfg.position_polyorder
    )
    wid_window, wid_poly = legalize_savgol_window(
        n_finite, cfg.width_window, cfg.width_polyorder
    )
    applied["position_window"] = pos_window
    applied["position_polyorder"] = pos_poly
    applied["width_window"] = wid_window
    applied["width_polyorder"] = wid_poly
    applied["orientation_sigma_frames"] = cfg.orientation_sigma_frames
    applied["orientation_kernel_size"] = min(
        cfg.orientation_kernel_size if cfg.orientation_kernel_size % 2 == 1 else cfg.orientation_kernel_size + 1,
        T if T % 2 == 1 else max(1, T - 1),
    )

    out_position = np.array(position, dtype=np.float64, copy=True)
    out_width = np.array(width, dtype=np.float64, copy=True)
    out_rotation = np.array(rotation, dtype=np.float64, copy=True)

    idx = np.nonzero(finite)[0]
    contiguous = idx.size == T
    if contiguous:
        out_position = _savgol(position, pos_window, pos_poly)
        out_width = _savgol(width[:, None], wid_window, wid_poly)[:, 0]
        finite_rot = np.isfinite(rotation).all(axis=(1, 2))
        if finite_rot.all():
            out_rotation = _gaussian_slerp_sequence(
                rotation,
                sigma=cfg.orientation_sigma_frames,
                kernel_size=cfg.orientation_kernel_size,
            )
        else:
            applied["rotation_skipped"] = "some frames have no finite orientation"
    else:
        # Smooth only the concatenated valid run so a gap cannot leak motion into
        # the filtered signal; invalid frames keep their previous values.
        sub_pos = position[idx]
        sub_wid = width[idx]
        sub_rot = rotation[idx]
        out_position[idx] = _savgol(sub_pos, *legalize_savgol_window(len(idx), cfg.position_window, cfg.position_polyorder))
        out_width[idx] = _savgol(
            sub_wid[:, None], *legalize_savgol_window(len(idx), cfg.width_window, cfg.width_polyorder)
        )[:, 0]
        if np.isfinite(sub_rot).all():
            out_rotation[idx] = _gaussian_slerp_sequence(
                sub_rot,
                sigma=cfg.orientation_sigma_frames,
                kernel_size=cfg.orientation_kernel_size,
            )
        applied["gapped_input"] = True

    return SmoothingResult(out_position, out_rotation, out_width, applied)
