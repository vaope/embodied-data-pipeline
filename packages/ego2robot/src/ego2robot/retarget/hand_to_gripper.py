"""Hand-to-gripper retargeting: MANO-21 keypoints -> canonical ``T_C_TCP``.

Implements system-design.md section 7 and the paper's equations 1-3 and
appendix A.3 verbatim::

    p_vf  = 0.7 * p_index + 0.3 * p_middle                   (paper eq. 1)
    p_tcp = (p_thumb + p_vf) / 2,  w = ||p_thumb - p_vf||    (paper eq. 2)
    z     = s * (p_thumb - p_vf) / w
    y     = normalize(cross(z, d)),  d = p_vf - p_wrist
    x     = cross(y, z)
    R_C_TCP = [x y z]                                        (paper eq. 3)

with ``s = +1`` for the right hand and ``s = -1`` for the left.

Two things this module refuses to do:

* fabricate an orientation.  When the frame degenerates and no previous valid
  orientation exists, the frame is marked invalid instead of being filled with
  the identity (system-design.md 7, [ENGINEERING]).
* emit a non-orthonormal matrix.  Every rotation goes through
  :func:`~ego2robot.core.transforms.so3_repair`, and the correction magnitude is
  persisted so drift is visible rather than absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.frames import Frame, TransformName, transform_name
from ..core.schema import ARRAY_SPECS, CanonicalActionSequence, HandSequence
from ..core.transforms import (
    matrix_from_quat_xyzw,
    quat_xyzw_from_matrix,
    rotation_matrix_from_axes,
    so3_repair,
)

__all__ = [
    "MANO_INDEX_MIDDLE_TIP",
    "MANO_INDEX_TIP",
    "MANO_THUMB_TIP",
    "MANO_WRIST",
    "FallbackCode",
    "RetargetConfigLike",
    "RetargetResult",
    "retarget_hand_sequence",
    "tcp_frame_from_keypoints",
]

# MANO-21 indices (see ego2robot.ingest.egodex.MANO21_JOINT_NAMES)
MANO_WRIST = 0
MANO_THUMB_TIP = 4
MANO_INDEX_TIP = 8
MANO_MIDDLE_TIP = 12

_RETARGET_INDICES = (MANO_WRIST, MANO_THUMB_TIP, MANO_INDEX_TIP, MANO_MIDDLE_TIP)

#: Names of the four joints the retargeting maths consumes, in order.
RETARGET_JOINT_NAMES = ("wrist", "thumb_tip", "index_tip", "middle_tip")


class FallbackCode:
    """Stable bit flags for per-frame retargeting reasons (system-design.md 6.6).

    Kept as plain integers so they can be persisted as ``uint16`` and extended
    without changing the schema.
    """

    NONE = 0
    HAND_INVALID = 1 << 0
    NON_FINITE_KEYPOINT = 1 << 1
    WIDTH_BELOW_THRESHOLD = 1 << 2
    JAW_PLANE_DEGENERATE = 1 << 3
    FROZEN_TO_PREVIOUS = 1 << 4
    NO_PREVIOUS_ORIENTATION = 1 << 5
    SO3_REPAIRED = 1 << 6
    NON_FINITE_INPUT = 1 << 7

    _NAMES = {
        NONE: "none",
        HAND_INVALID: "hand_invalid",
        NON_FINITE_KEYPOINT: "non_finite_keypoint",
        WIDTH_BELOW_THRESHOLD: "width_below_threshold",
        JAW_PLANE_DEGENERATE: "jaw_plane_degenerate",
        FROZEN_TO_PREVIOUS: "frozen_to_previous",
        NO_PREVIOUS_ORIENTATION: "no_previous_orientation",
        SO3_REPAIRED: "so3_repaired",
        NON_FINITE_INPUT: "non_finite_input",
    }

    @classmethod
    def decode(cls, code: int) -> list[str]:
        if code == 0:
            return ["none"]
        return [name for bit, name in cls._NAMES.items() if bit and (code & bit)]


@dataclass
class RetargetConfigLike:
    """The subset of :class:`~ego2robot.core.config.RetargetConfig` used here."""

    virtual_fingertip_index_weight: float = 0.7
    virtual_fingertip_middle_weight: float = 0.3
    handedness_right_sign: int = 1
    handedness_left_sign: int = -1
    min_gripper_width_m: float = 0.01
    parallel_dot_threshold: float = 0.999
    repair_so3: bool = True
    max_so3_correction_rad: float = 1e-3


@dataclass
class RetargetResult:
    """Per-frame canonical gripper output plus audit trail."""

    position_C_m: NDArray[np.float64]  # [T, 3]
    rotation_C_TCP: NDArray[np.float64]  # [T, 3, 3]
    gripper_width_m: NDArray[np.float64]  # [T]
    confidence: NDArray[np.float32]  # [T]
    valid: NDArray[np.bool_]  # [T]
    fallback_code: NDArray[np.uint16]  # [T]
    so3_correction_rad: NDArray[np.float32]  # [T]
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def T_C_TCP(self) -> NDArray[np.float64]:
        T = np.tile(np.eye(4, dtype=np.float64), (self.position_C_m.shape[0], 1, 1))
        T[:, :3, :3] = self.rotation_C_TCP
        T[:, :3, 3] = self.position_C_m
        return T

    @property
    def quaternion_C_xyzw(self) -> NDArray[np.float64]:
        return np.stack([quat_xyzw_from_matrix(R) for R in self.rotation_C_TCP], axis=0)


def tcp_frame_from_keypoints(
    keypoints_C_m: NDArray[np.float64],
    *,
    handedness: int,
    cfg: RetargetConfigLike | None = None,
    previous_R_C_TCP: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64], int, float]:
    """Retarget a single frame.

    Returns ``(R_C_TCP, gripper_width_m, tcp_position_C_m, fallback_code,
    so3_correction_rad)``.

    ``R_C_TCP`` is ``NaN`` when the frame cannot produce an orientation and no
    previous estimate exists.  The **position** is returned independently of the
    orientation degeneracy: the paper freezes the *orientation* when the jaw
    closes past 10 mm, but ``p_tcp = (p_thumb + p_vf) / 2`` stays perfectly well
    defined, and freezing the position too would teleport the gripper back to the
    last open-jaw location.
    """
    cfg = cfg or RetargetConfigLike()
    kp = np.asarray(keypoints_C_m, dtype=np.float64)
    if kp.shape != (21, 3):
        raise ValueError(f"keypoints must be [21, 3], got {kp.shape}")

    code = FallbackCode.NONE
    selected = kp[list(_RETARGET_INDICES)]
    if not np.all(np.isfinite(selected)):
        code |= FallbackCode.NON_FINITE_KEYPOINT
        return _frozen_or_nan(previous_R_C_TCP, code), float("nan"), np.full(3, np.nan), code, 0.0

    p_wrist, p_thumb, p_index, p_middle = selected
    w_idx = cfg.virtual_fingertip_index_weight
    w_mid = cfg.virtual_fingertip_middle_weight
    p_vf = w_idx * p_index + w_mid * p_middle
    p_tcp = 0.5 * (p_thumb + p_vf)
    w = float(np.linalg.norm(p_thumb - p_vf))
    d = p_vf - p_wrist

    sign = cfg.handedness_right_sign if handedness > 0 else cfg.handedness_left_sign
    if handedness == 0:
        raise ValueError(
            "handedness 0 (unknown) cannot define the grasp axis sign; "
            "the adapter must resolve left/right before retargeting"
        )

    degeneracies: list[int] = []
    if not np.isfinite(w) or w < cfg.min_gripper_width_m:
        degeneracies.append(FallbackCode.WIDTH_BELOW_THRESHOLD)

    z = np.full(3, np.nan)
    if not degeneracies:
        z = sign * (p_thumb - p_vf) / w
        # ||z x d|| ~ 0 means the jaw line and the wrist->fingertip direction are
        # parallel, so the jaw plane has no unique normal.
        sin_angle = float(np.linalg.norm(np.cross(z, d)))
        if sin_angle < np.sqrt(max(0.0, 1.0 - cfg.parallel_dot_threshold**2)):
            degeneracies.append(FallbackCode.JAW_PLANE_DEGENERATE)

    if degeneracies:
        for bit in degeneracies:
            code |= bit
        code |= FallbackCode.FROZEN_TO_PREVIOUS
        if previous_R_C_TCP is None:
            code |= FallbackCode.NO_PREVIOUS_ORIENTATION
            return np.full((3, 3), np.nan), w, p_tcp, code, 0.0
        return np.asarray(previous_R_C_TCP, dtype=np.float64).copy(), w, p_tcp, code, 0.0

    y_raw = np.cross(z, d)
    y_norm = float(np.linalg.norm(y_raw))
    if not np.isfinite(y_norm) or y_norm < 1e-12:
        code |= FallbackCode.JAW_PLANE_DEGENERATE | FallbackCode.FROZEN_TO_PREVIOUS
        if previous_R_C_TCP is None:
            code |= FallbackCode.NO_PREVIOUS_ORIENTATION
            return np.full((3, 3), np.nan), w, p_tcp, code, 0.0
        return np.asarray(previous_R_C_TCP, dtype=np.float64).copy(), w, p_tcp, code, 0.0

    y = y_raw / y_norm
    x = np.cross(y, z)
    R_raw = rotation_matrix_from_axes(x, y, z)

    if not cfg.repair_so3:
        return R_raw, w, p_tcp, code, 0.0

    repair = so3_repair(R_raw)
    if repair.corrected:
        code |= FallbackCode.SO3_REPAIRED
    return repair.R, w, p_tcp, code, float(repair.correction_rad)


def _frozen_or_nan(
    previous_R_C_TCP: NDArray[np.float64] | None, code: int
) -> NDArray[np.float64]:
    if previous_R_C_TCP is None:
        return np.full((3, 3), np.nan)
    return np.asarray(previous_R_C_TCP, dtype=np.float64).copy()


def retarget_hand_sequence(
    sequence: HandSequence,
    *,
    hand_index: int = 0,
    cfg: RetargetConfigLike | None = None,
) -> tuple[RetargetResult, CanonicalActionSequence]:
    """Retarget every frame of a :class:`HandSequence`.

    No temporal filtering happens here: this is the *raw* canonical output.
    Outlier rejection, gap filling and smoothing live in their own modules so
    that the raw, filtered and smoothed trajectories can all be kept for
    comparison (implementation-plan.md Gate M2).
    """
    cfg = cfg or RetargetConfigLike()
    kp = np.asarray(sequence["keypoints_C_m"], dtype=np.float64)[:, hand_index]
    valid_in = np.asarray(sequence["valid"])[:, hand_index]
    conf = np.asarray(sequence["detection_confidence"])[:, hand_index].astype(np.float32)
    handedness = np.asarray(sequence["handedness"])[:, hand_index]
    T = kp.shape[0]

    position = np.full((T, 3), np.nan, dtype=np.float64)
    rotation = np.full((T, 3, 3), np.nan, dtype=np.float64)
    width = np.full(T, np.nan, dtype=np.float64)
    codes = np.zeros(T, dtype=np.uint16)
    corrections = np.zeros(T, dtype=np.float32)
    valid = np.zeros(T, dtype=bool)

    previous_R: NDArray[np.float64] | None = None
    frozen_count = 0
    repaired_count = 0
    degenerate_count = 0

    for t in range(T):
        if not bool(valid_in[t]):
            codes[t] = FallbackCode.HAND_INVALID
            continue
        heading = int(handedness[t])
        R, w, p_tcp, code, corr = tcp_frame_from_keypoints(
            kp[t], handedness=heading, cfg=cfg, previous_R_C_TCP=previous_R
        )
        codes[t] = code
        corrections[t] = corr
        width[t] = w
        if code & FallbackCode.FROZEN_TO_PREVIOUS:
            frozen_count += 1
        if code & FallbackCode.JAW_PLANE_DEGENERATE:
            degenerate_count += 1
        if code & FallbackCode.SO3_REPAIRED:
            repaired_count += 1
        # the position is valid whenever the four retarget joints are finite, even
        # if the orientation had to be frozen (see tcp_frame_from_keypoints)
        if np.all(np.isfinite(p_tcp)):
            position[t] = p_tcp
        if code & FallbackCode.NO_PREVIOUS_ORIENTATION or not np.all(np.isfinite(R)):
            continue
        rotation[t] = R
        valid[t] = True
        previous_R = R

    diagnostics = {
        "frames": float(T),
        "valid_frames": float(np.count_nonzero(valid)),
        "frozen_frames": float(frozen_count),
        "degenerate_frames": float(degenerate_count),
        "so3_repaired_frames": float(repaired_count),
        "so3_max_correction_rad": float(np.max(corrections)) if T else 0.0,
    }

    result = RetargetResult(
        position_C_m=position,
        rotation_C_TCP=rotation,
        gripper_width_m=width,
        confidence=conf,
        valid=valid,
        fallback_code=codes,
        so3_correction_rad=corrections,
        diagnostics=diagnostics,
    )

    tcp_name = transform_name(Frame.C, Frame.TCP)
    position_C_m = np.asarray(result.position_C_m, dtype=np.float64)
    quats = _quaternion_series(result.rotation_C_TCP)
    transforms = _transform_series(result.rotation_C_TCP, position_C_m)
    action = CanonicalActionSequence(
        {
            "timestamp_s": np.asarray(sequence["timestamp_s"], dtype=np.float64),
            "frame_index": np.asarray(sequence["frame_index"], dtype=np.int64),
            "T_C_TCP": transforms[:, None, :, :],
            "tcp_position_C_m": position_C_m[:, None, :],
            "tcp_quaternion_C_xyzw": quats[:, None, :],
            "gripper_width_m": result.gripper_width_m[:, None],
            "confidence": result.confidence.reshape(T, 1),
            "valid": result.valid.reshape(T, 1),
            "fallback_code": result.fallback_code.reshape(T, 1),
            "so3_correction_rad": result.so3_correction_rad.reshape(T, 1),
        },
        meta={
            "pose_variable": str(tcp_name),
            "axes": {"x": "approach", "y": "gripper normal (jaw-plane normal)", "z": "grasp axis"},
            "handedness": int(handedness[0]) if T else 0,
            "handedness_sign": cfg.handedness_right_sign if (T and handedness[0] > 0) else cfg.handedness_left_sign,
            "invalid_frame_policy": (
                "a frame that cannot produce an orientation stays NaN in this raw table and "
                "carries a non-zero fallback_code plus valid=False; downstream gap handling "
                "fills it and records that it did so"
            ),
            "raw_output": True,
        },
    )
    return result, action


def _quaternion_series(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """``[T, 4]`` xyzw quaternions, NaN on rows without a finite rotation."""
    q = np.full((rotation.shape[0], 4), np.nan, dtype=np.float64)
    ok = np.isfinite(rotation).all(axis=(1, 2))
    for t in np.nonzero(ok)[0]:
        q[t] = quat_xyzw_from_matrix(rotation[t])
    return q


def _transform_series(
    rotation: NDArray[np.float64], position: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``[T, 4, 4]`` transforms with a valid homogeneous bottom row."""
    T = rotation.shape[0]
    out = np.full((T, 4, 4), np.nan, dtype=np.float64)
    ok = np.isfinite(rotation).all(axis=(1, 2))
    out[ok, :3, :3] = rotation[ok]
    out[ok, :3, 3] = position[ok]
    out[ok, 3, :] = (0.0, 0.0, 0.0, 1.0)
    return out
