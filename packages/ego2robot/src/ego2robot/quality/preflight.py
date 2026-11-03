"""Preflight: input-integrity checks that run before any expensive stage.

[ENGINEERING] system-design.md 10.1 — decode success, monotonic timestamps,
valid ``K``, no unexpected NaN, schema/shape agreement.  Preflight deliberately
does **not** change algorithmic semantics; it only refuses to spend GPU or IK
time on inputs that cannot produce a valid product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core import artifacts as A
from ..core.schema import ValidationResult

__all__ = ["REASON_CODES", "PreflightReport", "run_preflight"]


class REASON_CODES:
    """Episode and frame level reason codes (system-design.md 6.6).

    Frame level codes are also used as bit flags by the retargeter; episode
    level codes are stable strings so a report stays readable across versions.
    """

    # episode level
    EPISODE_VALID = "episode_valid"
    NO_VIDEO = "no_video"
    NO_HAND_ANNOTATIONS = "no_hand_annotations"
    TIME_NOT_MONOTONIC = "time_not_monotonic"
    CAMERA_INVALID = "camera_invalid"
    TIME_LENGTH_MISMATCH = "time_length_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    NON_FINITE_DATA = "non_finite_data"
    HAND_ALL_INVALID = "hand_all_invalid"
    HAND_INVALID_RATIO = "hand_invalid_ratio"
    IK_SUCCESS_LOW = "ik_success_low"
    FK_ERROR_HIGH = "fk_error_high"
    ROBOT_INVISIBLE = "robot_invisible"
    SELF_COLLISION = "self_collision"
    DEPTH_INVALID = "depth_invalid"
    CAMERA_ASSUMED = "camera_assumed_debug_only"


#: Reasons that must halt the pipeline rather than merely annotate the report.
BLOCKING_REASONS: frozenset[str] = frozenset(
    {
        REASON_CODES.NO_VIDEO,
        REASON_CODES.NO_HAND_ANNOTATIONS,
        REASON_CODES.TIME_NOT_MONOTONIC,
        REASON_CODES.CAMERA_INVALID,
        REASON_CODES.TIME_LENGTH_MISMATCH,
        REASON_CODES.SCHEMA_MISMATCH,
        REASON_CODES.NON_FINITE_DATA,
        REASON_CODES.HAND_ALL_INVALID,
    }
)


@dataclass
class PreflightReport:
    """Outcome of the preflight sweep."""

    episode_id: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    checked: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "ego2robot.preflight.v1",
            "episode_id": self.episode_id,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "checked": list(self.checked),
        }


def run_preflight(layout: A.EpisodeLayout, *, episode_id: str = "") -> PreflightReport:
    """Run every cheap integrity check over whatever the episode already has."""
    from ..core.schema import CanonicalActionSequence, EpisodeManifest, HandSequence, RobotTrajectory

    report = PreflightReport(episode_id=episode_id or layout.root.name, passed=True)

    # ---- manifest ---------------------------------------------------------
    if not layout.episode_json.is_file():
        report.reasons.append(REASON_CODES.NO_VIDEO)
        report.passed = False
        return report
    try:
        episode = EpisodeManifest.model_validate(A.read_json(layout.episode_json))
    except Exception as exc:  # noqa: BLE001 - reported as a reason code
        report.reasons.append(REASON_CODES.SCHEMA_MISMATCH)
        report.metrics["episode_manifest_error"] = f"{type(exc).__name__}: {exc}"
        report.passed = False
        return report
    report.checked.append("episode_manifest")

    video = layout.root / episode.source.video_uri
    if not video.is_file():
        report.reasons.append(REASON_CODES.NO_VIDEO)

    ts = episode.timebase.timestamps_s
    if ts is not None:
        arr = np.asarray(ts, dtype=np.float64)
        if arr.size and not np.all(np.diff(arr) > 0.0):
            report.reasons.append(REASON_CODES.TIME_NOT_MONOTONIC)
        if arr.size != episode.timebase.num_frames:
            report.reasons.append(REASON_CODES.TIME_LENGTH_MISMATCH)
        report.metrics["num_frames"] = int(episode.timebase.num_frames)
        report.metrics["duration_s"] = round(float(arr[-1] - arr[0]), 6) if arr.size > 1 else 0.0
    report.checked.append("timebase")

    try:
        K = episode.camera.K_array
        ok_K = (
            K.shape == (3, 3)
            and np.isfinite(K).all()
            and K[0, 0] > 0
            and K[1, 1] > 0
            and K[2, 2] == 1.0
        )
    except Exception:  # noqa: BLE001
        ok_K = False
    if not ok_K:
        report.reasons.append(REASON_CODES.CAMERA_INVALID)
    if episode.camera.calibration_source == "assumed":
        report.reasons.append(REASON_CODES.CAMERA_ASSUMED)
    report.metrics["camera_calibration_source"] = episode.camera.calibration_source
    report.checked.append("camera")

    # ---- hand -------------------------------------------------------------
    if layout.hand_npz.is_file():
        try:
            hand = HandSequence.load_npz(layout.hand_npz)
        except Exception as exc:  # noqa: BLE001
            report.reasons.append(REASON_CODES.SCHEMA_MISMATCH)
            report.metrics["hand_error"] = f"{type(exc).__name__}: {exc}"
            hand = None
        if hand is not None:
            check = hand.validate()
            if not check.ok:
                report.reasons.append(REASON_CODES.SCHEMA_MISMATCH)
                report.metrics["hand_schema_errors"] = [i.message for i in check.errors]
            valid = np.asarray(hand["valid"])
            ratio = float(np.mean(valid)) if valid.size else 0.0
            report.metrics["hand_valid_ratio"] = round(ratio, 6)
            if ratio == 0.0:
                report.reasons.append(REASON_CODES.HAND_ALL_INVALID)
            kp = np.asarray(hand["keypoints_C_m"])
            non_finite = int(np.count_nonzero(~np.isfinite(kp)))
            report.metrics["hand_non_finite_keypoints"] = non_finite
            if ts is not None and len(ts) != hand.num_frames:
                report.reasons.append(REASON_CODES.TIME_LENGTH_MISMATCH)
                report.metrics["hand_frames"] = hand.num_frames
            report.checked.append("hand")
    else:
        report.reasons.append(REASON_CODES.NO_HAND_ANNOTATIONS)

    # ---- canonical --------------------------------------------------------
    if layout.actions_npz.is_file():
        action = CanonicalActionSequence.load_npz(layout.actions_npz)
        check = action.validate()
        if not check.ok:
            report.reasons.append(REASON_CODES.SCHEMA_MISMATCH)
            report.metrics["canonical_schema_errors"] = [i.message for i in check.errors]
        valid = np.asarray(action["valid"])[:, 0]
        report.metrics["canonical_valid_ratio"] = round(float(np.mean(valid)), 6) if valid.size else 0.0
        report.checked.append("canonical")

    # ---- robot ------------------------------------------------------------
    if layout.trajectory_npz.is_file():
        traj = RobotTrajectory.load_npz(layout.trajectory_npz)
        success = np.asarray(traj["ik_success"])[:, 0]
        ratio = float(np.mean(success)) if success.size else 0.0
        pos_err = np.asarray(traj["position_error_m"])[:, 0]
        rot_err = np.asarray(traj["rotation_error_rad"])[:, 0]
        finite = np.isfinite(pos_err) & np.isfinite(rot_err)
        p95_pos = float(np.percentile(pos_err[finite], 95)) if finite.any() else float("inf")
        p95_rot = float(np.percentile(rot_err[finite], 95)) if finite.any() else float("inf")
        report.metrics.update(
            {
                "ik_success_ratio": round(ratio, 6),
                "p95_position_error_m": round(p95_pos, 9) if np.isfinite(p95_pos) else None,
                "p95_rotation_error_rad": round(p95_rot, 9) if np.isfinite(p95_rot) else None,
            }
        )
        if ratio < 0.95:
            report.reasons.append(REASON_CODES.IK_SUCCESS_LOW)
        report.checked.append("robot")

    report.passed = not any(r in BLOCKING_REASONS for r in report.reasons)
    return report
