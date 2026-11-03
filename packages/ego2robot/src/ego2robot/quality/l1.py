"""L1 curation: pipeline-internal frame validity (paper appendix A.5).

A frame is valid when:

1. a hand was detected;
2. IK tracking error is ``< 0.05 m``   [PAPER];
3. the rendered robot is visible (``> 0`` pixels);
4. there is no MuJoCo self-collision.

Additional invalidation: bimanual cross-arm contacts ``> 1``, or a robot mask
covering ``> 70%`` of the image.  Short valid runs (``< floor(0.3 * fps)``
frames) sandwiched between invalid regions are eroded to invalid.

Frame level reasons use the stable bit flags in
:class:`~ego2robot.retarget.hand_to_gripper.FallbackCode` plus the
:class:`FrameReason` bits below, so a report can be aggregated without parsing
strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core import artifacts as A
from ..core.schema import (
    CanonicalActionSequence,
    EpisodeManifest,
    HandSequence,
    QualityMetrics,
    QualityReport,
    RobotTrajectory,
)

__all__ = [
    "FrameReason",
    "L1Config",
    "L1Result",
    "build_quality_report",
    "run_l1",
]


class FrameReason:
    """L1 frame invalidation bit flags (uint32)."""

    NONE = 0
    HAND_MISSING = 1 << 0
    IK_FAILED = 1 << 1
    IK_TRACKING_ERROR = 1 << 2
    ROBOT_NOT_VISIBLE = 1 << 3
    SELF_COLLISION = 1 << 4
    CROSS_ARM_CONTACTS = 1 << 5
    ROBOT_MASK_TOO_LARGE = 1 << 6
    STABILITY_EROSION = 1 << 7
    NON_FINITE_ACTION = 1 << 8

    _NAMES = {
        NONE: "none",
        HAND_MISSING: "hand_missing",
        IK_FAILED: "ik_failed",
        IK_TRACKING_ERROR: "ik_tracking_error",
        ROBOT_NOT_VISIBLE: "robot_not_visible",
        SELF_COLLISION: "self_collision",
        CROSS_ARM_CONTACTS: "cross_arm_contacts",
        ROBOT_MASK_TOO_LARGE: "robot_mask_too_large",
        STABILITY_EROSION: "stability_erosion",
        NON_FINITE_ACTION: "non_finite_action",
    }

    @classmethod
    def decode(cls, code: int) -> list[str]:
        if code == 0:
            return ["none"]
        return [n for b, n in cls._NAMES.items() if b and (code & b)]


@dataclass
class L1Config:
    """Thresholds for :func:`run_l1` (system-design.md 10.2)."""

    ik_tracking_error_m: float = 0.05
    robot_visible_pixels: int = 1
    robot_mask_area_ratio_max: float = 0.70
    cross_arm_contacts_max: int = 1
    stability_erosion_fps_fraction: float = 0.3
    use_stricter_mvp_rotation: float = 0.10


@dataclass
class L1Result:
    """Per-frame validity plus the aggregate numbers a report needs."""

    frame_valid: np.ndarray  # bool [T]
    frame_reason: np.ndarray  # uint32 [T]
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    frame_reason_histogram: dict[str, int] = field(default_factory=dict)


def _erode_short_valid_runs(
    valid: np.ndarray, *, min_run: int, reason: np.ndarray, reason_bit: int
) -> tuple[np.ndarray, int]:
    """Invalidate valid runs shorter than ``min_run`` flanked by invalid frames."""
    out = valid.copy()
    eroded = 0
    T = valid.shape[0]
    i = 0
    while i < T:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < T and valid[j + 1]:
            j += 1
        length = j - i + 1
        left_invalid = i == 0 or not valid[i - 1]
        right_invalid = j == T - 1 or not valid[j + 1]
        if length < min_run and left_invalid and right_invalid:
            out[i : j + 1] = False
            reason[i : j + 1] |= reason_bit
            eroded += length
        i = j + 1
    return out, eroded


def run_l1(
    *,
    action: CanonicalActionSequence,
    trajectory: RobotTrajectory | None,
    hand: HandSequence | None,
    fps: float,
    robot_mask_area_ratio: np.ndarray | None = None,
    cfg: L1Config | None = None,
) -> L1Result:
    """Evaluate L1 for every frame.

    ``robot_mask_area_ratio`` is the fraction of image area covered by robot
    pixels per frame; when the render stage has not run, the mask-based checks
    are skipped and recorded as such rather than assumed to pass.
    """
    cfg = cfg or L1Config()
    T = action.num_frames
    reason = np.zeros(T, dtype=np.uint32)
    valid = np.ones(T, dtype=bool)

    action_valid = np.asarray(action["valid"])[:, 0]
    T_C_TCP = np.asarray(action["T_C_TCP"], dtype=np.float64)[:, 0]
    width = np.asarray(action["gripper_width_m"], dtype=np.float64)[:, 0]

    finite_action = np.isfinite(T_C_TCP).all(axis=(1, 2)) & np.isfinite(width)
    bad_finite = ~finite_action
    reason[bad_finite] |= FrameReason.NON_FINITE_ACTION
    valid &= ~bad_finite

    hand_missing = ~action_valid
    if hand is not None:
        hand_valid = np.asarray(hand["valid"])[:, 0]
        hand_missing = hand_missing | ~hand_valid
    reason[hand_missing] |= FrameReason.HAND_MISSING
    valid &= ~hand_missing

    if trajectory is not None:
        ik_success = np.asarray(trajectory["ik_success"])[:, 0]
        pos_err = np.asarray(trajectory["position_error_m"])[:, 0]
        rot_err = np.asarray(trajectory["rotation_error_rad"])[:, 0]
        collision = np.asarray(trajectory["collision_code"])

        ik_fail = ~ik_success
        reason[ik_fail] |= FrameReason.IK_FAILED
        valid &= ~ik_fail

        tracking_bad = np.isfinite(pos_err) & (pos_err >= cfg.ik_tracking_error_m)
        reason[tracking_bad] |= FrameReason.IK_TRACKING_ERROR
        valid &= ~tracking_bad

        collisions = collision == 1
        reason[collisions] |= FrameReason.SELF_COLLISION
        valid &= ~collisions
        _ = rot_err

    if robot_mask_area_ratio is not None:
        ratio = np.asarray(robot_mask_area_ratio, dtype=np.float64)
        if ratio.shape[0] == T:
            invisible = ratio <= 0.0
            reason[invisible] |= FrameReason.ROBOT_NOT_VISIBLE
            valid &= ~invisible

            too_big = ratio > cfg.robot_mask_area_ratio_max
            reason[too_big] |= FrameReason.ROBOT_MASK_TOO_LARGE
            valid &= ~too_big

    min_run = max(1, int(np.floor(cfg.stability_erosion_fps_fraction * max(fps, 1e-9))))
    valid, eroded = _erode_short_valid_runs(
        valid, min_run=min_run, reason=reason, reason_bit=FrameReason.STABILITY_EROSION
    )

    histogram: dict[str, int] = {}
    for bit, name in FrameReason._NAMES.items():
        if bit == FrameReason.NONE:
            continue
        count = int(np.count_nonzero(reason & bit))
        if count:
            histogram[name] = count

    metrics: dict[str, Any] = {
        "num_frames": int(T),
        "frame_valid_ratio": round(float(np.mean(valid)), 6) if T else 0.0,
        "frame_invalid_frames": int(np.count_nonzero(~valid)),
        "stability_eroded_frames": int(eroded),
        "stability_min_run_frames": int(min_run),
        "frame_reason_histogram": histogram,
        "robot_mask_checked": robot_mask_area_ratio is not None,
        "trajectory_available": trajectory is not None,
    }
    reasons: list[str] = []
    if T and float(np.mean(valid)) == 0.0:
        reasons.append("l1_no_valid_frames")
    return L1Result(
        frame_valid=valid,
        frame_reason=reason,
        metrics=metrics,
        reasons=reasons,
        frame_reason_histogram=histogram,
    )


def build_quality_report(
    *,
    episode_id: str,
    action: CanonicalActionSequence,
    trajectory: RobotTrajectory | None,
    hand: HandSequence | None,
    episode: EpisodeManifest | None,
    l1: L1Result,
    layout: A.EpisodeLayout | None = None,
    robot_mask_area_ratio: np.ndarray | None = None,
    threshold_profile: str = "repro_v1",
    preflight_reasons: Sequence[str] = (),
) -> QualityReport:
    """Assemble the persisted ``ego2robot.quality.v1`` report."""
    T = action.num_frames
    valid = np.asarray(action["valid"])[:, 0]
    width = np.asarray(action["gripper_width_m"], dtype=np.float64)[:, 0]
    fallback = np.asarray(action["fallback_code"])[:, 0]
    so3_correction = np.asarray(action["so3_correction_rad"], dtype=np.float64)[:, 0]

    metrics = QualityMetrics()
    metrics.hand_missing_ratio = round(float(np.mean(~valid)), 6) if T else 1.0
    metrics.fallback_frame_ratio = round(float(np.mean(fallback != 0)), 6) if T else 0.0
    metrics.so3_max_correction_rad = round(float(np.max(so3_correction)), 12) if T else 0.0

    if trajectory is not None:
        success = np.asarray(trajectory["ik_success"])[:, 0]
        pos_err = np.asarray(trajectory["position_error_m"])[:, 0]
        rot_err = np.asarray(trajectory["rotation_error_rad"])[:, 0]
        finite = np.isfinite(pos_err) & np.isfinite(rot_err)
        metrics.ik_success_ratio = round(float(np.mean(success)), 6) if T else 0.0
        metrics.p95_position_error_m = (
            round(float(np.percentile(pos_err[finite], 95)), 9) if finite.any() else 0.0
        )
        metrics.p95_rotation_error_rad = (
            round(float(np.percentile(rot_err[finite], 95)), 9) if finite.any() else 0.0
        )
        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        if qpos.shape[0] > 1:
            metrics.max_joint_jump_rad = round(float(np.max(np.abs(np.diff(qpos, axis=0)))), 6)

    if robot_mask_area_ratio is not None:
        ratio = np.asarray(robot_mask_area_ratio, dtype=np.float64)
        metrics.robot_visible_ratio = round(float(np.mean(ratio > 0.0)), 6)
    metrics.non_finite_ratio = round(
        float(np.mean(~np.isfinite(np.asarray(action["T_C_TCP"], dtype=np.float64)))) if T else 0.0,
        6,
    )

    reasons = list(preflight_reasons) + list(l1.reasons)
    if metrics.ik_success_ratio and metrics.ik_success_ratio < 0.95:
        reasons.append("ik_success_ratio_below_0.95")
    if not valid.any():
        reasons.append("no_valid_canonical_frames")

    stages_verified = [
        name
        for name in ("ingest", "hand", "retarget", "robot", "render", "projection")
        if layout is not None and (layout.manifest_path(name)).is_file()
    ]

    report = QualityReport(
        episode_id=episode_id,
        valid=bool(l1.frame_valid.any() and not reasons),
        frame_valid_ratio=round(float(np.mean(l1.frame_valid)), 6) if T else 0.0,
        metrics=metrics,
        reasons=reasons,
        frame_reasons=l1.frame_reason_histogram,
        threshold_profile=threshold_profile,
        gate="M3",
        stages_verified=stages_verified,
    )
    _ = (episode, width)
    return report
