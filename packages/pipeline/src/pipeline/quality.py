"""Persist real per-frame engineering QC and explicit training exclusions.

The existing Ego2Robot L1 implementation is reused unchanged.  Additional
checks close its missing finite-residual/rotation/missing-input cases.  This
module performs no perception, retargeting, IK or model inference.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .actions import ACTION_PROFILE, CAMERA_FRAME_SEMANTICS, LEGACY_ACTION_PROFILE, build_transition_arrays

RULESET_ID = "learning.stage5.engineering_qc.v1"
REASON_REGISTRY_VERSION = "learning.stage5.reason_bits.v1"
_REASON_NAMES = (
    "S5_SOURCE_INVALID", "S5_HAND_MISSING", "S5_CANONICAL_INVALID",
    "S5_IK_FAILED", "S5_IK_TRACKING_ERROR", "S5_IK_ROTATION_ERROR",
    "S5_SELF_COLLISION", "S5_ROBOT_NOT_VISIBLE", "S5_ROBOT_MASK_TOO_LARGE",
    "S5_STABILITY_EROSION", "S5_NON_FINITE_ACTION", "S5_GRIPPER_SATURATED",
    "S5_VISUAL_MISSING", "S5_VISUAL_PREVIEW_ONLY", "S5_DEPTH_NOT_METRIC",
    "S5_QC_NOT_EVALUATED", "S5_ADMISSION_UNREVIEWED", "S5_ADMISSION_REJECTED",
    "S5_MOCK_ARTIFACT", "S5_REVIEW_SCOPE_MISMATCH", "S5_ACTION_PROFILE_UNRESOLVED",
    "S5_JOINT_LIMIT", "S5_SOURCE_LENGTH_MISMATCH", "S5_TIMEBASE_MISMATCH",
    "S5_ARTIFACT_INVALID", "S5_CANONICAL_REPAIRED", "S5_GRIPPER_BINDING_MISMATCH",
    "S5_CAMERA_MOTION_NOT_EVALUATED", "S5_CROSS_ARM_CONTACTS",
)
REASON_BITS = {name: 1 << bit for bit, name in enumerate(_REASON_NAMES)}


def decode_reason_bits(value: int) -> list[str]:
    return [name for name, bit in REASON_BITS.items() if int(value) & bit]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path, source_key: str, schema: str, format_: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size,
            "format": format_, "schema_version": schema, "source_key": source_key,
            "producer": RULESET_ID, "availability": "available"}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record(path: Path | None, schema: str, ts: np.ndarray, shapes: dict[str, tuple[int, ...]]) -> tuple[dict | None, dict]:
    info: dict[str, Any] = {"availability": "missing", "path": str(path) if path else None}
    if path is None or not path.is_file():
        return None, info
    info.update(sha256=_sha256(path), bytes=path.stat().st_size)
    try:
        with np.load(path, allow_pickle=False) as stored:
            data = {key: stored[key].copy() for key in stored.files if not key.startswith("__")}
            version = str(stored["__schema_version__"].item())
            meta = json.loads(str(stored["__meta_json__"].item())) if "__meta_json__" in stored else {}
        if version != schema:
            raise ValueError(f"schema mismatch: {version!r} != {schema!r}")
        n = len(ts)
        for key, trailing in shapes.items():
            if key not in data or data[key].shape != (n, *trailing):
                raise ValueError(f"{key} expected {(n, *trailing)}, got {getattr(data.get(key), 'shape', None)}")
        if not np.array_equal(data["frame_index"], np.arange(n)):
            raise ValueError("frame_index differs from source 0..N-1")
        if not np.allclose(data["timestamp_s"], ts, atol=1e-6, rtol=0, equal_nan=False):
            raise ValueError("timestamp_s differs from source PTS (tolerance 1e-6 s)")
        info.update(availability="available", schema_version=version, meta=meta)
        return data, info
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        info.update(availability="invalid", error=str(error))
        return None, info


def _camera_motion(path: Path, ts: np.ndarray, out: Path) -> dict[str, Any]:
    """Quantify real T_W_C; deliberately make no invented motion cutoff."""
    result: dict[str, Any] = {"status": "not_evaluated", "source": str(path),
        "camera_frame_semantics": CAMERA_FRAME_SEMANTICS,
        "motion_compensated": False, "threshold": None,
        "use_scope_decision": "engineering_camera_relative_view_only"}
    try:
        with h5py.File(path, "r") as fh:
            transform = np.asarray(fh["transforms/camera"], dtype=np.float64)
        if transform.shape != (len(ts), 4, 4) or not np.isfinite(transform).all():
            raise ValueError("T_W_C must be finite [N,4,4], matching source frame count")
        rot = transform[:, :3, :3]
        if not (np.max(np.abs(np.swapaxes(rot, 1, 2) @ rot - np.eye(3))) < 1e-4 and
                np.max(np.abs(np.linalg.det(rot) - 1)) < 1e-4):
            raise ValueError("T_W_C rotation is not SO(3)")
        translation = transform[:, :3, 3]
        delta_m = np.linalg.norm(np.diff(translation, axis=0), axis=1)
        delta_rad = np.linalg.norm(Rotation.from_matrix(rot[1:] @ np.swapaxes(rot[:-1], 1, 2)).as_rotvec(), axis=1) if len(ts) > 1 else np.zeros(0)
        offset_m = np.linalg.norm(translation - translation[0], axis=1) if len(ts) else np.zeros(0)
        offset_rad = np.linalg.norm(Rotation.from_matrix(rot @ rot[0].T).as_rotvec(), axis=1) if len(ts) else np.zeros(0)
        np.savez_compressed(out, __schema_version__=np.asarray("learning.stage5.camera_motion.v1"), frame_index=np.arange(len(ts), dtype=np.int64), timestamp_s=ts,
            T_W_C=transform, delta_translation_m=delta_m, delta_rotation_rad=delta_rad,
            translation_from_first_m=offset_m, rotation_from_first_rad=offset_rad)
        def stats(values: np.ndarray) -> dict:
            return {"count": int(values.size), "max": float(values.max()) if values.size else 0.0,
                    "p95": float(np.percentile(values, 95)) if values.size else 0.0,
                    "mean": float(values.mean()) if values.size else 0.0}
        result.update(status="executed", transform="T_W_C: camera to ARKit origin",
            source_sha256=_sha256(path), delta_translation_m=stats(delta_m), delta_rotation_rad=stats(delta_rad),
            translation_from_first_m=stats(offset_m), rotation_from_first_rad=stats(offset_rad),
            artifact_path=str(out.resolve()), note="Measured head motion is reported without inferring a safe physical-control threshold.")
    except (KeyError, OSError, ValueError) as error:
        result["error"] = str(error)
    return result


def _mask_ratios(root: Path | None, n: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    ratio, evaluated = np.full(n, np.nan), np.zeros(n, dtype=bool)
    refs = []
    if root is None:
        return ratio, evaluated, refs
    for i in range(n):
        arm_path = root / "masks" / "robot" / f"frame-{i:06d}.png"
        gripper_path = root / "masks" / "gripper" / f"frame-{i:06d}.png"
        if not arm_path.is_file() or not gripper_path.is_file():
            continue
        try:
            with Image.open(arm_path) as image:
                arm = np.asarray(image).copy()
            with Image.open(gripper_path) as image:
                grip = np.asarray(image).copy()
            if arm.ndim != 2 or grip.shape != arm.shape:
                raise ValueError("robot/gripper masks must be same-size single-channel images")
            if not np.isin(np.unique(arm), [0, 1, 255]).all() or not np.isin(np.unique(grip), [0, 1, 255]).all():
                raise ValueError("mask values must be bool/0/1/255")
            ratio[i] = float(np.mean((arm > 0) | (grip > 0)))
            evaluated[i] = True
            refs.extend({"path": str(p.resolve()), "sha256": _sha256(p), "bytes": p.stat().st_size, "frame_index": i,
                         "mask_kind": kind} for p, kind in ((arm_path, "arm_body"), (gripper_path, "gripper")))
        except (ValueError, OSError):
            continue
    return ratio, evaluated, refs


def _visual_state(index: dict | None, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Accept only a real, non-preview metric-depth composite index."""
    n = len(ts)
    valid, reason = np.zeros(n, dtype=bool), np.zeros(n, dtype=np.uint64)
    rows = (index or {}).get("frames", (index or {}).get("records", []))
    if isinstance(rows, dict):
        rows = rows.get("depth_composite_rgb", [])
    by_frame: dict[int, list[dict]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("image_kind") == "depth_composite_rgb":
            try:
                by_frame.setdefault(int(row.get("frame_index", -1)), []).append(row)
            except (TypeError, ValueError):
                continue
    for i in range(n):
        matches = by_frame.get(i, [])
        if len(matches) != 1:
            reason[i] |= np.uint64(REASON_BITS["S5_VISUAL_MISSING"])
            continue
        row = matches[0]
        if row.get("preview_only", True):
            reason[i] |= np.uint64(REASON_BITS["S5_VISUAL_PREVIEW_ONLY"])
        depth_mode = row.get("depth_mode", (index or {}).get("depth_mode"))
        if depth_mode not in {"metric", "aligned_metric"}:
            reason[i] |= np.uint64(REASON_BITS["S5_DEPTH_NOT_METRIC"])
        evidence_kind = row.get("evidence_kind", "unavailable")
        if evidence_kind in {"mock", "synthetic_fixture", "unavailable"}:
            reason[i] |= np.uint64(REASON_BITS["S5_MOCK_ARTIFACT"])
        if not row.get("valid", False) or row.get("metric_valid") is False:
            reason[i] |= np.uint64(REASON_BITS["S5_VISUAL_MISSING"])
        try:
            if abs(float(row["timestamp_s"]) - ts[i]) > 1e-6:
                reason[i] |= np.uint64(REASON_BITS["S5_TIMEBASE_MISMATCH"])
            path = Path(row["path"])
            if not path.is_file() or _sha256(path) != row.get("sha256"):
                reason[i] |= np.uint64(REASON_BITS["S5_ARTIFACT_INVALID"])
        except (KeyError, TypeError, ValueError, OSError):
            reason[i] |= np.uint64(REASON_BITS["S5_ARTIFACT_INVALID"])
        valid[i] = reason[i] == 0
    # Readiness is independent from file presence: a pending M5/M6 review is
    # never promoted by the compositor's successful program execution.
    gates_record = (index or {}).get("gates", {})
    reviews_passed = (index or {}).get("required_reviews_passed") is True or (
        bool(gates_record) and all(gates_record.get(name, {}).get("status") in {"accepted", "passed"}
                                   for name in ("m3", "m5", "m6")))
    gates = [] if reviews_passed else ["required_visual_reviews_pending_or_not_provided"]
    return valid, reason, gates


def build_quality_and_actions(
    *, episode_root: Path | None, source_hdf5: Path, timestamp_s: np.ndarray,
    source_frame_valid: np.ndarray, source_key: str, output_dir: Path,
    visual_index: dict | None, admission: dict, annotations: dict | None = None,
) -> dict[str, Any]:
    """Write quality/frames.npz, quality/report.json and adjacent action view.

    This strict engineering profile excludes saturated endpoints, unresolved
    action bindings, missing reviews and missing real composite observations.
    It still writes inspectable actions and geometric candidate masks.
    """
    output_dir, source_hdf5 = Path(output_dir), Path(source_hdf5)
    episode_root = Path(episode_root) if episode_root is not None else None
    ts = np.asarray(timestamp_s, dtype=np.float64)
    source_valid = np.asarray(source_frame_valid, dtype=bool)
    if ts.ndim != 1 or source_valid.shape != ts.shape or not len(ts) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError("QC requires nonempty finite strictly increasing source PTS and bool[N] source validity")
    n = len(ts)
    qdir, vdir = output_dir / "quality", output_dir / "views"
    qdir.mkdir(parents=True, exist_ok=True)
    vdir.mkdir(parents=True, exist_ok=True)
    bit = REASON_BITS
    reason = np.zeros(n, dtype=np.uint64)
    def mark(mask: np.ndarray | bool, code: str) -> None:
        reason[mask] |= np.uint64(bit[code])
    def path(relative: str) -> Path | None:
        return episode_root / relative if episode_root is not None else None
    base = {"frame_index": (), "timestamp_s": ()}
    hand, hand_info = _record(path("hand/sequence.npz"), "ego2robot.hand.v1", ts,
        {**base, "valid": (1,), "keypoints_C_m": (1, 21, 3), "handedness": (1,), "detection_confidence": (1,)})
    canonical, canonical_info = _record(path("canonical/actions.npz"), "ego2robot.canonical_action.v1", ts,
        {**base, "valid": (1,), "T_C_TCP": (1, 4, 4), "tcp_position_C_m": (1, 3),
         "tcp_quaternion_C_xyzw": (1, 4), "gripper_width_m": (1,), "fallback_code": (1,)})
    robot, robot_info = _record(path("robot/franka_panda/trajectory.npz"), "ego2robot.robot_trajectory.v1", ts,
        {**base, "ik_success": (1,), "qpos": (7,), "qvel": (7,), "gripper_qpos": (2,),
         "position_error_m": (1,), "rotation_error_rad": (1,), "collision_code": (),
         "T_B_TCP_target": (1, 4, 4), "T_B_TCP_fk": (1, 4, 4), "joint_limit_hit": (), "solver_status": ()})
    inputs = {"hand": hand_info, "canonical": canonical_info, "robot": robot_info}
    for info in inputs.values():
        if info["availability"] == "invalid":
            mark(True, "S5_ARTIFACT_INVALID")
            if "timestamp" in info.get("error", "") or "frame_index" in info.get("error", ""):
                mark(True, "S5_TIMEBASE_MISMATCH")
            else:
                mark(True, "S5_SOURCE_LENGTH_MISMATCH")
        meta_text = json.dumps(info.get("meta", {})).lower()
        if '"mock"' in meta_text or '"synthetic_fixture"' in meta_text or '"mock/' in meta_text:
            mark(True, "S5_MOCK_ARTIFACT")
    hand_source_valid = hand["valid"][:, 0].astype(bool) if hand is not None else np.zeros(n, bool)
    observed = hand_source_valid.copy()
    if hand is not None:
        observed &= np.isfinite(hand["keypoints_C_m"][:, 0, [0, 4, 8, 12], :]).all(axis=(1, 2))
        observed &= hand["handedness"][:, 0] == 1
    mark(~source_valid, "S5_SOURCE_INVALID")
    mark(~observed, "S5_HAND_MISSING")
    transform = canonical["T_C_TCP"][:, 0] if canonical is not None else None
    requested = canonical["gripper_width_m"][:, 0] if canonical is not None else np.full(n, np.nan)
    computed = canonical["valid"][:, 0].astype(bool) if canonical is not None else np.zeros(n, bool)
    canonical_valid = computed.copy()
    fallback = canonical["fallback_code"][:, 0] if canonical is not None else np.zeros(n, np.uint16)
    repaired = np.zeros(n, bool)
    if canonical is not None:
        r = transform[:, :3, :3]
        finite = np.isfinite(transform).all(axis=(1, 2)) & np.isfinite(requested) & (requested >= 0)
        good = np.flatnonzero(finite)
        geometry = np.zeros(n, bool)
        if good.size:
            geometry[good] = (np.max(np.abs(np.swapaxes(r[good], 1, 2) @ r[good] - np.eye(3)), axis=(1, 2)) <= 1e-5) & (np.abs(np.linalg.det(r[good]) - 1) <= 1e-5)
        geometry &= np.all(np.isclose(transform[:, 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0), axis=1)
        quaternion = canonical["tcp_quaternion_C_xyzw"][:, 0]
        quaternion_ok = np.isfinite(quaternion).all(axis=1) & (np.abs(np.linalg.norm(quaternion, axis=1) - 1) <= 1e-5)
        good_quaternion = np.flatnonzero(quaternion_ok)
        if good_quaternion.size:
            quaternion_ok[good_quaternion] &= np.max(np.abs(Rotation.from_quat(quaternion[good_quaternion]).as_matrix() - r[good_quaternion]), axis=(1, 2)) <= 1e-5
        geometry &= quaternion_ok
        geometry &= np.all(np.isclose(canonical["tcp_position_C_m"][:, 0], transform[:, :3, 3], atol=1e-6, rtol=0), axis=1)
        canonical_valid &= finite & geometry
        repaired = (fallback.astype(np.uint64) & np.uint64(~64 & 0xFFFF)) != 0
        filtered, filtered_info = _record(path("canonical/actions-filtered.npz"), "ego2robot.canonical_action.v1", ts, {**base, "valid": (1,)})
        inputs["canonical_filtered"] = filtered_info
        if filtered is not None:
            repaired |= computed & ~filtered["valid"][:, 0]
    mark(~canonical_valid, "S5_CANONICAL_INVALID")
    mark(repaired, "S5_CANONICAL_REPAIRED")
    saturated = np.isfinite(requested) & ((requested < 0.0) | (requested > 0.08))
    mark(saturated, "S5_GRIPPER_SATURATED")
    ik_valid = np.zeros(n, bool)
    bound = np.full(n, np.nan)
    binding_mismatch = np.zeros(n, bool)
    if robot is not None:
        meta = robot_info.get("meta", {})
        bound = np.sum(robot["gripper_qpos"], axis=1)
        binding_known = meta.get("robot_id") == "franka_panda" and meta.get("gripper_width_to_qpos") == "finger_joint = width_m / 2"
        if not binding_known:
            bound[:] = np.nan
            mark(True, "S5_ACTION_PROFILE_UNRESOLVED")
        finger = robot["gripper_qpos"]
        binding_mismatch = ~np.isfinite(bound) | (np.abs(bound - np.clip(requested, 0, 0.08)) > 1e-7)
        binding_mismatch |= (finger < -1e-7).any(axis=1) | (finger > 0.04 + 1e-7).any(axis=1)
        binding_mismatch |= np.abs(finger[:, 0] - finger[:, 1]) > 1e-7
        ik_valid = robot["ik_success"][:, 0].astype(bool)
        residual_finite = np.isfinite(robot["position_error_m"][:, 0]) & np.isfinite(robot["rotation_error_rad"][:, 0])
        residual_finite &= np.isfinite(robot["qpos"]).all(axis=1) & np.isfinite(robot["qvel"]).all(axis=1)
        residual_finite &= np.isfinite(robot["T_B_TCP_target"]).all(axis=(1, 2, 3)) & np.isfinite(robot["T_B_TCP_fk"]).all(axis=(1, 2, 3))
        ik_valid &= residual_finite & (robot["solver_status"] == 0)
        mark((robot["position_error_m"][:, 0] >= 0.05) | ~residual_finite, "S5_IK_TRACKING_ERROR")
        mark((robot["rotation_error_rad"][:, 0] >= 0.10) | ~residual_finite, "S5_IK_ROTATION_ERROR")
        mark(robot["collision_code"] == 1, "S5_SELF_COLLISION")
        mark(robot["joint_limit_hit"].astype(bool), "S5_JOINT_LIMIT")
        ik_valid &= (robot["position_error_m"][:, 0] < 0.05) & (robot["rotation_error_rad"][:, 0] < 0.10)
        ik_valid &= (robot["collision_code"] != 1) & ~robot["joint_limit_hit"].astype(bool)
    mark(~ik_valid, "S5_IK_FAILED")
    mark(binding_mismatch, "S5_GRIPPER_BINDING_MISMATCH")
    ratio, mask_evaluated, mask_refs = _mask_ratios(episode_root, n)
    mark(mask_evaluated & (ratio <= 0), "S5_ROBOT_NOT_VISIBLE")
    mark(mask_evaluated & (ratio > 0.70), "S5_ROBOT_MASK_TOO_LARGE")
    l1_valid, l1_reason = np.zeros(n, bool), np.zeros(n, np.uint32)
    l1_info: dict[str, Any] = {"status": "not_evaluated", "reason": "required hand/canonical/robot input missing or invalid"}
    if hand is not None and canonical is not None and robot is not None:
        # Repository dependency location is derived from the supplied episode,
        # never from an unrelated same-named file discovered by globbing.
        repo_src = episode_root.parents[2] / "src"
        if str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
        from ego2robot.core.schema import CanonicalActionSequence, HandSequence, RobotTrajectory
        from ego2robot.quality.l1 import FrameReason, run_l1
        fps = 1.0 / float(np.median(np.diff(ts))) if n > 1 else 1.0
        # NaN masks preserve unknowns.  They never count as complete coverage.
        l1 = run_l1(action=CanonicalActionSequence(canonical), trajectory=RobotTrajectory(robot),
                    hand=HandSequence(hand), fps=fps, robot_mask_area_ratio=ratio)
        l1_valid, l1_reason = l1.frame_valid, l1.frame_reason
        l1_info = {"status": "executed", "implementation": "ego2robot.quality.l1.run_l1",
            "implementation_sha256": _sha256(repo_src / "ego2robot/quality/l1.py"),
            "metrics": l1.metrics, "notes": "Original L1 rotation and missing-mask gaps are separately guarded by Stage 5."}
        mark((l1_reason & FrameReason.STABILITY_EROSION) != 0, "S5_STABILITY_EROSION")
    camera = _camera_motion(source_hdf5, ts, qdir / "camera-motion.npz")
    if camera["status"] == "executed":
        camera["artifact_ref"] = _ref(qdir / "camera-motion.npz", source_key, "learning.stage5.camera_motion.v1", "npz")
    if camera["status"] != "executed":
        mark(True, "S5_CAMERA_MOTION_NOT_EVALUATED")
    # This is an explicit engineering view, whose camera-relative semantics
    # require no new manual gate.  It is not a claim about real robot control.
    # A missing or incompatible canonical/Panda binding remains unresolved.
    profile_resolved = bool(canonical is not None and robot is not None and np.isfinite(bound).all())
    if not profile_resolved:
        mark(True, "S5_ACTION_PROFILE_UNRESOLVED")
    visual_valid, visual_reasons, review_gaps = _visual_state(visual_index, ts)
    reason |= visual_reasons
    if review_gaps:
        mark(True, "S5_REVIEW_SCOPE_MISMATCH")
    admission_status = str(admission.get("status", admission.get("decision", "not_evaluated")))
    if admission_status == "reject":
        mark(True, "S5_ADMISSION_REJECTED")
    elif admission_status != "pass":
        mark(True, "S5_ADMISSION_UNREVIEWED")
    evaluated = np.full(n, hand is not None and canonical is not None and robot is not None, dtype=bool) & mask_evaluated
    evaluated &= camera["status"] == "executed"
    mark(~evaluated, "S5_QC_NOT_EVALUATED")
    geometry_valid = source_valid & observed & canonical_valid & ik_valid & l1_valid & mask_evaluated
    geometry_valid &= (ratio > 0) & (ratio <= 0.70) & ~saturated & ~repaired & ~binding_mismatch
    fatal_geometry = bit["S5_MOCK_ARTIFACT"] | bit["S5_ARTIFACT_INVALID"] | bit["S5_SOURCE_LENGTH_MISMATCH"] | bit["S5_TIMEBASE_MISMATCH"]
    geometry_valid &= (reason & np.uint64(fatal_geometry)) == 0
    qc_valid = geometry_valid & visual_valid & evaluated & (reason == 0)
    transition = build_transition_arrays(timestamp_s=ts, frame_valid=qc_valid, frame_reason_bits=reason,
        T_C_TCP=transform, requested_width_m=requested, bound_width_m=bound,
        non_finite_reason_bit=bit["S5_NON_FINITE_ACTION"], geometry_frame_valid=geometry_valid)
    frames_path, transitions_path = qdir / "frames.npz", vdir / "transitions.npz"
    np.savez_compressed(frames_path, __schema_version__=np.asarray("learning.stage5.quality_frames.v1"), frame_index=np.arange(n, dtype=np.int64), timestamp_s=ts,
        source_valid=source_valid, hand_source_valid=hand_source_valid, hand_observed_valid=observed,
        canonical_computed_valid=computed, canonical_valid=canonical_valid, canonical_repaired=repaired,
        canonical_fallback_code=fallback, ik_valid=ik_valid, visual_valid=visual_valid,
        gripper_saturated=saturated, requested_gripper_width_m=requested, bound_gripper_width_m=bound,
        gripper_binding_mismatch=binding_mismatch, robot_mask_area_ratio=ratio, robot_mask_evaluated=mask_evaluated,
        qc_evaluated=evaluated, qc_frame_valid=qc_valid, geometry_candidate_valid=geometry_valid,
        l1_frame_valid=l1_valid, l1_reason_bits=l1_reason, reason_bits=reason)
    np.savez_compressed(transitions_path, __schema_version__=np.asarray("learning.stage5.transitions.v1"), **transition)
    counts = {"frames": n, "transitions": n - 1, "source_valid": int(source_valid.sum()),
        "hand_observed_valid": int(observed.sum()), "canonical_valid": int(canonical_valid.sum()),
        "ik_valid": int(ik_valid.sum()), "visual_valid": int(visual_valid.sum()),
        "qc_evaluated": int(evaluated.sum()), "qc_frame_valid": int(qc_valid.sum()),
        "gripper_saturated": int(saturated.sum()), "geometry_candidate_frames": int(geometry_valid.sum()),
        "geometry_candidate_transitions": int(transition["geometry_candidate_valid"].sum()),
        "training_valid_transitions": int(transition["transition_valid"].sum())}
    coverage = {
        "source_validity": {"status": "executed", "evaluated_frames": n},
        "canonical_finite_so3_and_original_hand": {"status": "executed" if canonical is not None and hand is not None else "not_evaluated"},
        "existing_l1": l1_info,
        "robot_residual_finite_rotation_joint_limit": {"status": "executed" if robot is not None else "not_evaluated", "position_error_m_lt": 0.05, "rotation_error_rad_lt": 0.10},
        "robot_mask_union_area": {"status": "executed" if mask_evaluated.all() else "partial" if mask_evaluated.any() else "not_evaluated", "evaluated_frames": int(mask_evaluated.sum()), "range": "0 < ratio <= 0.70", "meaning": "union of separate arm_body/gripper masks"},
        "gripper_binding": {"status": "executed" if robot is not None else "not_evaluated", "robot_id": "franka_panda", "range_m": [0, 0.08], "policy": "reject transitions with either saturated endpoint", "bound_width_source": "sum of two recorded finger qpos"},
        "cross_arm_contacts": {"status": "not_applicable", "reason": "single right arm profile; no bimanual contact stream"},
        "real_metric_composite": {"status": "executed", "valid_frames": int(visual_valid.sum())},
        "camera_motion": camera,
        "L2": {"status": "deferred", "reason": "no independent dataset statistics and accepted threshold profile; three integration examples do not calibrate Q1/Q99"},
        "L3": {"status": "deferred", "backend": "manual_review", "reviewer": None, "consistent": None},
    }
    report = {"schema_version": "learning.stage5.quality_report.v1", "ruleset_id": RULESET_ID,
        "source_key": source_key, "threshold_profile": "path_a_right_franka_engineering_mvp.v1",
        "paper_full_L1_L2_L3_implemented": False, "reason_registry_version": REASON_REGISTRY_VERSION,
        "reason_registry": {name: {"bit": int(bit_value.bit_length() - 1), "value": bit_value, "mask": bit_value} for name, bit_value in bit.items()},
        "counts": counts, "rule_coverage": coverage, "input_refs": inputs, "mask_refs": mask_refs,
        "frame_reason_counts": {name: int(np.count_nonzero(reason & np.uint64(flag))) for name, flag in bit.items()},
        "transition_reason_counts": {name: int(np.count_nonzero(transition["reason_bits"] & np.uint64(flag))) for name, flag in bit.items()},
        "training_ready": bool(transition["transition_valid"].any() and admission_status == "pass"),
        "reason_codes": [name for name, flag in bit.items() if np.any(reason & np.uint64(flag))],
        "episode_decision": "training_ready" if transition["transition_valid"].any() and admission_status == "pass" else "candidate_only", "training_readiness": "training_ready" if transition["transition_valid"].any() and admission_status == "pass" else "candidate_only",
        "review_scope": {"required_visual_reviews": "pending" if review_gaps else "provided"},
        "action_profile": {"view_id": ACTION_PROFILE, "legacy_view_id": LEGACY_ACTION_PROFILE,
            "dtype": "float32", "shape": [n - 1, 7], "camera_frame_semantics": CAMERA_FRAME_SEMANTICS,
            "resolved_for_camera_view": profile_resolved, "physical_robot_control_claim": False,
            "delta_translation_unit": "m/transition", "delta_rotation_unit": "rad/transition",
            "rotation_composition": "Log(R[t+1] @ R[t].T)", "gripper_channel": "bound target width_m[t+1]",
            "legacy_gripper_channel": "requested canonical width_m[t]", "source_time_resampled": False},
        "admission_status": admission_status,
        "annotation_context": annotations or {"availability": "not_provided"},
        "notes": ["No inference or IK was run. Existing artifacts are read-only.",
            "The engineering checks were executed where inputs exist; L2/L3 are deferred, not pass.",
            "Finite qpos is never a substitute for recorded IK success and finite FK residuals.",
            "Candidate geometry counts do not represent training eligibility or human acceptance."]}
    report_path = qdir / "report.json"
    _json(report_path, report)
    return {"quality_ref": _ref(frames_path, source_key, "learning.stage5.quality_frames.v1", "npz"),
        "quality_report_ref": _ref(report_path, source_key, "learning.stage5.quality_report.v1", "json"),
        "transition_ref": _ref(transitions_path, source_key, "learning.stage5.transitions.v1", "npz"),
        "report": report, "counts": counts, "action_profile": report["action_profile"], "camera_motion": camera,
        "camera_motion_ref": camera.get("artifact_ref"), "training_readiness": report["training_readiness"]}
