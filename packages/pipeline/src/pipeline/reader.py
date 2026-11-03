"""Strict, read-only consumer for learning Stage 5 local-reference bundles.

This custom format is deliberately not a LeRobot/RLDS dataset.  Opening in
``inspect`` mode preserves missing evidence and invalid transitions.  The
default training mode never upgrades candidate or preview artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator

import cv2
import numpy as np

from .supervision import STRICT_STEPS, SOURCE_TASK, SUPERVISION_PROFILES, build_supervision

BUNDLE_SCHEMA = "learning.stage5.bundle.v1"
ACTION_PROFILE = "canonical_delta_C_target_width_m.v1"
TIME_TOLERANCE_S = 1e-6


class BundleValidationError(ValueError):
    """An artifact, schema, identity, or frame association is inconsistent."""


class TrainingReadinessError(BundleValidationError):
    """The requested training sample does not satisfy its declared gates."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_sha256(image: np.ndarray) -> str:
    """Hash decoded RGB uint8 pixels, in row-major order (not PNG bytes)."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise BundleValidationError("RGB pixels must be uint8[H,W,3]")
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise BundleValidationError(f"Expected JSON object: {path}")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _ref(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("availability") in {
        "missing", "not_evaluated", "deferred", "skipped_by_user_hold"
    }:
        return None
    if "path" in value and "sha256" in value:
        return value
    for key in ("npz_ref", "sequence_ref", "actions_ref", "trajectory_ref", "ref"):
        if key in value:
            found = _ref(value[key])
            if found:
                return found
    return None


class BundleReader:
    """Validated bundle with N observations and exactly N-1 adjacent slots.

    ``len(reader)`` includes invalid slots in inspect mode.  They retain their
    original indices; filtering must never join across a dropped frame.
    """

    def __init__(self, path: str | Path, mode: str = "train") -> None:
        if mode not in {"train", "inspect"}:
            raise ValueError("mode must be 'train' or 'inspect'")
        supplied = Path(path).resolve()
        self.bundle_path = supplied / "bundle.json" if supplied.is_dir() else supplied
        self.root = self.bundle_path.parent
        self.mode = mode
        self.bundle = _json(self.bundle_path)
        self._checked: dict[str, tuple[str, int]] = {}
        self._video_capture: cv2.VideoCapture | None = None
        self._video_path: Path | None = None
        self._video_next = 0
        self._image_cache: dict[int, np.ndarray] = {}
        self._validate_bundle()

    def _path(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

    def _verify_ref(self, reference: dict[str, Any]) -> Path:
        path = self._path(reference["path"])
        wanted = str(reference.get("sha256", "")).lower()
        size = reference.get("bytes")
        if len(wanted) != 64 or any(c not in "0123456789abcdef" for c in wanted):
            raise BundleValidationError(f"Missing/invalid SHA256: {path}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BundleValidationError(f"Missing/invalid byte count: {path}")
        source_key = reference.get("source_key")
        if source_key is not None and source_key != self.bundle["source_key"]:
            raise BundleValidationError(f"Wrong source_key: {path}")
        previous = self._checked.get(str(path))
        if previous is not None:
            if previous != (wanted, size):
                raise BundleValidationError(f"Conflicting artifact references: {path}")
            return path
        if not path.is_file():
            raise BundleValidationError(f"Missing artifact: {path}")
        if path.stat().st_size != size or sha256_file(path) != wanted:
            raise BundleValidationError(f"S5_SOURCE_HASH_MISMATCH: {path}")
        self._checked[str(path)] = (wanted, size)
        return path

    def _verify_nested_refs(self, value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self._verify_ref(value)
            for nested in value.values():
                self._verify_nested_refs(nested)
        elif isinstance(value, list):
            for nested in value:
                self._verify_nested_refs(nested)

    def _npz(self, value: Any, name: str, schema: str | None = None,
             required: bool = True) -> dict[str, np.ndarray] | None:
        reference = _ref(value)
        if reference is None:
            if required:
                raise BundleValidationError(f"Missing {name} NPZ reference")
            return None
        path = self._verify_ref(reference)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        declared = arrays.get("__schema_version__")
        actual = str(declared.item()) if declared is not None and declared.shape == () else None
        if schema and actual != schema:
            raise BundleValidationError(f"Wrong {name} schema: {actual!r}, expected {schema}")
        if not schema and actual is not None and not actual.endswith(".v1"):
            raise BundleValidationError(f"Unsupported {name} schema: {actual}")
        return arrays

    def _sidecar(self, value: Any, name: str, required: bool = False) -> dict[str, Any] | None:
        reference = _ref(value)
        if reference is None:
            if required:
                raise BundleValidationError(f"Missing {name} reference")
            return None
        result = _json(self._verify_ref(reference))
        if result.get("source_key", self.bundle["source_key"]) != self.bundle["source_key"]:
            raise BundleValidationError(f"Wrong source_key in {name}")
        self._verify_nested_refs(result)
        return result

    def _array(self, arrays: dict[str, np.ndarray], name: str,
               shape: tuple[int, ...], dtype: str | None = None) -> np.ndarray:
        if name not in arrays:
            raise BundleValidationError(f"Missing array {name}")
        array = arrays[name]
        if array.shape != shape or (dtype and array.dtype != np.dtype(dtype)):
            raise BundleValidationError(
                f"{name}: got {array.shape}/{array.dtype}, expected {shape}/{dtype}")
        return array

    def _clock(self, arrays: dict[str, np.ndarray], name: str) -> None:
        indices = self._array(arrays, "frame_index", (self.num_frames,), "int64")
        times = self._array(arrays, "timestamp_s", (self.num_frames,), "float64")
        if not np.array_equal(indices, self.frame_index) or not np.allclose(
            times, self.timestamp_s, atol=TIME_TOLERANCE_S, rtol=0, equal_nan=False
        ):
            raise BundleValidationError(f"S5_TIMEBASE_MISMATCH: {name}")

    def _validate_bundle(self) -> None:
        bundle = self.bundle
        if bundle.get("schema_version") != BUNDLE_SCHEMA:
            raise BundleValidationError("Unsupported bundle schema")
        for name in ("run_id", "episode_id", "source_key", "storage_mode", "training_readiness"):
            if not isinstance(bundle.get(name), str) or not bundle[name]:
                raise BundleValidationError(f"Missing {name}")
        if bundle["storage_mode"] not in {"local_reference", "self_contained_copy"}:
            raise BundleValidationError("Unsupported storage_mode")
        if not isinstance(bundle.get("artifacts"), dict) or not bundle["artifacts"]:
            raise BundleValidationError("A bundle must declare artifacts with checksums")
        self._verify_nested_refs(bundle)
        if bundle.get("bundle_schema_ref") is not None:
            import jsonschema
            schema = self._sidecar(bundle["bundle_schema_ref"], "bundle schema", required=True)
            jsonschema.Draft202012Validator(schema).validate(bundle)
        self.episode_id = bundle["episode_id"]
        self.training_readiness = bundle["training_readiness"]
        if self.training_readiness not in {"training_ready", "candidate_only", "blocked"}:
            raise BundleValidationError("Unknown training_readiness")
        self.readiness_reasons = bundle.get("readiness_reasons", [])
        self.action_profile = bundle.get("action_profile", bundle.get("views", {}).get("action_profile"))
        if isinstance(self.action_profile, dict):
            self.action_profile = self.action_profile.get("view_id", self.action_profile.get("id"))
        self.timebase = self._npz(bundle.get("timebase"), "timebase", "learning.stage5.timebase.v1")
        assert self.timebase is not None
        if "frame_index" not in self.timebase or self.timebase["frame_index"].ndim != 1:
            raise BundleValidationError("Invalid timebase frame_index")
        self.num_frames = len(self.timebase["frame_index"])
        if self.num_frames < 2:
            raise BundleValidationError("At least two source frames are required")
        self.frame_index = self._array(self.timebase, "frame_index", (self.num_frames,), "int64")
        self.timestamp_s = self._array(self.timebase, "timestamp_s", (self.num_frames,), "float64")
        edges = self._array(self.timebase, "frame_edge_s", (self.num_frames + 1,), "float64")
        self._array(self.timebase, "source_frame_valid", (self.num_frames,), "bool")
        if not np.array_equal(self.frame_index, np.arange(self.num_frames, dtype=np.int64)):
            raise BundleValidationError("Source indices must be exactly 0..N-1")
        if (not np.isfinite(edges).all() or not np.isfinite(self.timestamp_s).all()
                or not (np.diff(edges) > 0).all()
                or not np.array_equal(edges[:-1], self.timestamp_s)):
            raise BundleValidationError("S5_TIMEBASE_MISMATCH: PTS/edge sequence")
        refs = bundle.get("refs", {})
        self.source_video_ref = _ref(refs.get("source_video"))
        self.source_hdf5_ref = _ref(refs.get("source_hdf5"))
        if not self.source_video_ref or not self.source_hdf5_ref:
            raise BundleValidationError("Both source MP4 and HDF5 identities are required")
        identity = hashlib.sha256((self.source_video_ref["sha256"] + ":" + self.source_hdf5_ref["sha256"]).encode("ascii")).hexdigest()
        if bundle["source_key"] != identity:
            raise BundleValidationError("source_key does not bind source MP4 and HDF5 SHA256")
        self._validate_source_clock()
        self.hand = self._npz(refs.get("hand"), "hand", "ego2robot.hand.v1", required=False)
        self.canonical = self._npz(refs.get("canonical"), "canonical",
                                   "ego2robot.canonical_action.v1", required=False)
        self.robot = self._npz(refs.get("robot"), "robot",
                               "ego2robot.robot_trajectory.v1", required=False)
        for name, arrays in (("hand", self.hand), ("canonical", self.canonical), ("robot", self.robot)):
            if arrays is not None:
                self._clock(arrays, name)
        self._validate_geometry()
        quality = bundle.get("quality", {})
        self.quality = self._npz(quality.get("frames_ref"), "quality", "learning.stage5.quality_frames.v1")
        assert self.quality is not None
        self._clock(self.quality, "quality")
        for name in ("source_valid", "hand_observed_valid", "canonical_valid", "ik_valid",
                     "visual_valid", "qc_evaluated", "qc_frame_valid", "gripper_saturated"):
            self._array(self.quality, name, (self.num_frames,), "bool")
        self._array(self.quality, "reason_bits", (self.num_frames,), "uint64")
        self.quality_report = self._sidecar(quality.get("report_ref"), "quality report", required=True)
        self.transitions = self._npz(bundle.get("views", {}).get("transitions_ref"), "transitions", "learning.stage5.transitions.v1")
        assert self.transitions is not None
        self._validate_transitions()
        annotations = bundle.get("annotations", {})
        self.stage3 = self._sidecar(annotations.get("stage3_ref"), "Stage3")
        self.supervision_profile = bundle.get("supervision_profile", STRICT_STEPS)
        if self.supervision_profile not in SUPERVISION_PROFILES:
            raise BundleValidationError("Unsupported supervision_profile")
        expected_supervision = build_supervision(self.supervision_profile, self.source_metadata["attributes"],
                                                self.source_hdf5_ref, self.stage3)
        self.supervision = self._sidecar(annotations.get("supervision_ref"), "supervision")
        if self.supervision is not None:
            expected = {"source_key": bundle["source_key"], **expected_supervision}
            if self.supervision != expected:
                raise BundleValidationError("Supervision differs from bound HDF5 attributes, profile, or preserved Stage3")
            if annotations.get("supervision_profile") != self.supervision_profile:
                raise BundleValidationError("Supervision profile header/sidecar mismatch")
        elif "supervision_profile" in bundle:
            raise BundleValidationError("Versioned supervision requires its verified sidecar")
        else:
            # Read compatibility for historical bundles, never silent training
            # downgrade: formal training still checks the derived strict profile.
            self.supervision = expected_supervision
        self.task_meta = self.supervision["selected_task"]
        if self.task_meta is None and self.mode == "inspect":
            self.task_meta = {"availability": "missing", "source": "stage3.v1",
                              "missing_reason": "S5_STAGE3_MISSING",
                              "source_task_diagnostic": self.supervision["source_task"]}
        self.step_map = self._sidecar(annotations.get("step_map_ref"), "step map") or {"steps": [], "frames": []}
        self.object_associations = self._sidecar(annotations.get("object_associations_ref"), "object associations") or {"associations": []}
        self._validate_annotations()
        visual = bundle.get("visual", {})
        self.visual_index = self._sidecar(visual.get("index_ref"), "visual index", required=True)
        assert self.visual_index is not None
        self.image_kind = visual.get("preferred_kind", self.visual_index.get("preferred_kind", "source_rgb"))
        self._validate_visual()
        if self.mode == "train":
            self._require_training_ready()

    def _validate_geometry(self) -> None:
        n = self.num_frames
        if self.hand is not None:
            for key, shape, dtype in (
                ("keypoints_C_m", (n, 1, 21, 3), "float32"),
                ("keypoints_uv_px", (n, 1, 21, 2), "float32"),
                ("handedness", (n, 1), "int8"),
                ("detection_confidence", (n, 1), "float32"), ("valid", (n, 1), "bool")
            ):
                self._array(self.hand, key, shape, dtype)
        if self.canonical is not None:
            for key, shape, dtype in (
                ("T_C_TCP", (n, 1, 4, 4), "float64"),
                ("tcp_position_C_m", (n, 1, 3), "float64"),
                ("tcp_quaternion_C_xyzw", (n, 1, 4), "float64"),
                ("gripper_width_m", (n, 1), "float64"), ("valid", (n, 1), "bool"),
                ("confidence", (n, 1), "float32"), ("fallback_code", (n, 1), "uint16")
            ):
                self._array(self.canonical, key, shape, dtype)
            valid = self.canonical["valid"][:, 0]
            transforms = self.canonical["T_C_TCP"][valid, 0]
            rotations = transforms[:, :3, :3]
            if (not np.isfinite(transforms).all()
                    or not np.allclose(transforms[:, 3, :], [0, 0, 0, 1], atol=1e-7, rtol=0)
                    or not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=1e-5, rtol=0)
                    or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)
                    or not np.allclose(transforms[:, :3, 3], self.canonical["tcp_position_C_m"][valid, 0], atol=1e-7, rtol=0)
                    or not np.allclose(np.linalg.norm(self.canonical["tcp_quaternion_C_xyzw"][valid, 0], axis=1), 1, atol=1e-5, rtol=0)):
                raise BundleValidationError("Invalid canonical rigid transform/position/quaternion")
        if self.robot is not None:
            for key, shape, dtype in (
                ("qpos", (n, 7), "float64"), ("qvel", (n, 7), "float64"),
                ("gripper_qpos", (n, 2), "float64"),
                ("T_B_TCP_target", (n, 1, 4, 4), "float64"),
                ("T_B_TCP_fk", (n, 1, 4, 4), "float64"),
                ("ik_success", (n, 1), "bool"), ("collision_code", (n,), "uint16"),
                ("joint_limit_hit", (n,), "bool")
            ):
                self._array(self.robot, key, shape, dtype)
            valid = self.robot["ik_success"][:, 0]
            if not np.isfinite(self.robot["qpos"][valid]).all() or not np.isfinite(self.robot["gripper_qpos"][valid]).all():
                raise BundleValidationError("Valid IK states contain non-finite joints")

    def _validate_source_clock(self) -> None:
        """Check the bound source, rather than trusting a copied timebase NPZ."""
        import h5py
        with h5py.File(self._path(self.source_hdf5_ref["path"]), "r") as source:
            camera = source.get("transforms/camera")
            if camera is None or camera.shape != (self.num_frames, 4, 4):
                raise BundleValidationError("S5_SOURCE_LENGTH_MISMATCH: EgoDex HDF5 camera transforms")
            self.source_metadata = {"attributes": _jsonable(dict(source.attrs)),
                                    "camera_intrinsic": _jsonable(np.asarray(source["camera/intrinsic"])) if "camera/intrinsic" in source else None,
                                    "source_video_ref": self.source_video_ref, "source_hdf5_ref": self.source_hdf5_ref}
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise BundleValidationError("ffprobe is required to verify original source PTS")
        command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
                   "-show_entries", "frame=best_effort_timestamp_time", "-of", "json",
                   str(self._path(self.source_video_ref["path"]))]
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True, timeout=60)
            frames = json.loads(result.stdout)["frames"]
            pts = np.asarray([float(frame["best_effort_timestamp_time"]) for frame in frames], dtype=np.float64)
        except (subprocess.SubprocessError, KeyError, ValueError) as error:
            raise BundleValidationError(f"Cannot verify original source PTS: {error}") from error
        if pts.shape != self.timestamp_s.shape:
            raise BundleValidationError("S5_SOURCE_LENGTH_MISMATCH: source video frame count")
        if not np.allclose(pts, self.timestamp_s, atol=TIME_TOLERANCE_S, rtol=0):
            raise BundleValidationError("S5_TIMEBASE_MISMATCH: original source video PTS")

    def _validate_transitions(self) -> None:
        assert self.transitions is not None
        arrays, count = self.transitions, self.num_frames - 1
        for key, dtype in (("source_frame_index", "int64"), ("target_frame_index", "int64"),
                           ("timestamp_s", "float64"), ("next_timestamp_s", "float64"),
                           ("dt_s", "float64"), ("transition_valid", "bool"), ("reason_bits", "uint64")):
            self._array(arrays, key, (count,), dtype)
        actions = self._array(arrays, "action7", (count, 7), "float32")
        if "legacy_action7" in arrays:
            self._array(arrays, "legacy_action7", (count, 7), "float32")
        if not np.array_equal(arrays["source_frame_index"], self.frame_index[:-1]) or not np.array_equal(
            arrays["target_frame_index"], self.frame_index[1:]
        ):
            raise BundleValidationError("Transitions must be original adjacent i -> i+1 pairs")
        for name, expected in (("timestamp_s", self.timestamp_s[:-1]),
                               ("next_timestamp_s", self.timestamp_s[1:]),
                               ("dt_s", np.diff(self.timestamp_s))):
            if not np.allclose(arrays[name], expected, atol=TIME_TOLERANCE_S, rtol=0):
                raise BundleValidationError(f"S5_TIMEBASE_MISMATCH: transition {name}")
        valid = arrays["transition_valid"]
        if not np.isfinite(actions[valid]).all():
            raise BundleValidationError("Valid transitions contain non-finite actions")
        if self.canonical is not None and self.robot is not None and self.action_profile == ACTION_PROFILE:
            expected = np.empty_like(actions)
            expected[:, :3] = np.diff(self.canonical["tcp_position_C_m"][:, 0], axis=0)
            rotations = self.canonical["T_C_TCP"][:, 0, :3, :3]
            for i in range(count):
                if np.isfinite(rotations[i:i+2]).all():
                    expected[i, 3:6] = cv2.Rodrigues(rotations[i+1] @ rotations[i].T)[0].reshape(3)
                else:
                    expected[i, 3:6] = np.nan
            expected[:, 6] = self.robot["gripper_qpos"][1:].sum(axis=1)
            if not np.allclose(actions, expected, atol=1e-6, rtol=1e-5, equal_nan=True):
                raise BundleValidationError("action7 does not match target-width camera-delta profile")
        assert self.quality is not None
        endpoint_valid = np.ones(self.num_frames, dtype=bool)
        for key in ("source_valid", "canonical_valid", "ik_valid", "visual_valid", "qc_evaluated", "qc_frame_valid"):
            endpoint_valid &= self.quality[key]
        endpoint_valid &= ~self.quality["gripper_saturated"]
        if np.any(valid & ~(endpoint_valid[:-1] & endpoint_valid[1:])):
            raise BundleValidationError("A valid transition has an invalid QC endpoint")

    def _validate_annotations(self) -> None:
        self._steps_by_id: dict[int, dict[str, Any]] = {}
        self._frame_steps: list[list[int]] = [[] for _ in range(self.num_frames)]
        if self.stage3 is not None:
            if self.stage3.get("episode_id") != self.episode_id or self.stage3.get("frame_count") != self.num_frames:
                raise BundleValidationError("Stage3 source episode/frame_count mismatch")
            if not isinstance(self.stage3.get("task"), dict) or not isinstance(self.stage3.get("steps"), list):
                raise BundleValidationError("Stage3 must preserve Task + flat Steps Schema v1")
            historical_ref = self.bundle.get("artifacts", {}).get("stage3/run_manifest")
            if historical_ref is not None:
                manifest = self._sidecar(historical_ref, "Stage3 historical source manifest", required=True)
                matches = [row for row in manifest.get("episodes", []) if row.get("episode_id") == self.episode_id]
                if len(matches) != 1 or matches[0].get("source_video_sha256") != self.source_video_ref["sha256"]:
                    raise BundleValidationError("Stage3 historical source video SHA256 mismatch")
            annotations = self.bundle.get("annotations", {})
            schema_ref = annotations.get("annotation_schema_ref", annotations.get("schema_ref", annotations.get("stage3_schema_ref")))
            if schema_ref is not None:
                import jsonschema
                schema = self._sidecar(schema_ref, "Stage3 Schema v1", required=True)
                jsonschema.validate(self.stage3, schema)
            for step in self.stage3["steps"]:
                if not isinstance(step, dict):
                    raise BundleValidationError("Malformed Stage3 step")
                start, end, sid = step.get("start_frame"), step.get("end_frame_exclusive"), step.get("step_id")
                if (not all(isinstance(x, int) and not isinstance(x, bool) for x in (start, end, sid))
                        or not 0 <= start < end <= self.num_frames or sid in self._steps_by_id):
                    raise BundleValidationError("Invalid Stage3 v1 step interval/id")
                self._steps_by_id[sid] = step
                for frame in range(start, end):
                    self._frame_steps[frame].append(sid)
        side_steps = self.step_map.get("steps", [])
        if self.stage3 is None and side_steps:
            raise BundleValidationError("Step sidecar invents absent Stage3 annotations")
        if self.stage3 is not None and len(side_steps) != len(self._steps_by_id):
            raise BundleValidationError("Step sidecar must preserve every original v1 step")
        for step in side_steps:
            sid = step.get("step_id")
            original = self._steps_by_id.get(sid)
            if original is None or any(step.get(k) != original.get(k) for k in ("start_frame", "end_frame_exclusive")):
                raise BundleValidationError("Step sidecar differs from unchanged Stage3 v1")
            for key, boundary in (("start_s", original["start_frame"]), ("end_s", original["end_frame_exclusive"])):
                if key in step and abs(float(step[key]) - self.timebase["frame_edge_s"][boundary]) > TIME_TOLERANCE_S:
                    raise BundleValidationError("Step sidecar time does not match source frame edges")
        seen: set[int] = set()
        for record in self.step_map.get("frames", []):
            frame = record.get("frame_index")
            if not isinstance(frame, int) or frame in seen or not 0 <= frame < self.num_frames:
                raise BundleValidationError("Invalid/duplicate step-map frame")
            if sorted(record.get("step_ids", [])) != sorted(self._frame_steps[frame]):
                raise BundleValidationError("Step frame association differs from source v1 intervals")
            seen.add(frame)
        if seen != set(range(self.num_frames)):
            raise BundleValidationError("Step-map sidecar must retain all N source frame records")

    def _validate_visual(self) -> None:
        self._images: dict[int, dict[str, Any]] = {}
        rgb_kinds = {"source_rgb", "robot_only_rgb", "arm_removed_background",
                     "projection_debug", "alpha_debug", "depth_composite_rgb"}
        mask_kinds = {"arm_body_mask", "gripper_mask", "human_arm_mask", "object_mask",
                      "visible_robot_mask", "occluded_gripper_mask"}
        for record in self.visual_index.get("frames", []):
            frame = record.get("frame_index")
            if not isinstance(frame, int) or not 0 <= frame < self.num_frames:
                raise BundleValidationError("Visual index frame out of bounds")
            if abs(float(record["timestamp_s"]) - self.timestamp_s[frame]) > TIME_TOLERANCE_S:
                raise BundleValidationError("Visual index PTS mismatch")
            kind = record.get("image_kind")
            if kind in rgb_kinds:
                if record.get("color_space") != "RGB":
                    raise BundleValidationError("Persisted RGB image boundaries must be RGB")
            elif kind in mask_kinds:
                if str(record.get("color_space", "")).lower() not in {"mask", "gray", "grey", "bool", "boolean"}:
                    raise BundleValidationError("Mask records must declare mask/gray/bool color space")
                if kind == self.image_kind:
                    raise BundleValidationError("The observation view must contain RGB images, not masks")
            else:
                raise BundleValidationError(f"Unsupported visual index image_kind: {kind}")
            if record.get("image_kind") == self.image_kind:
                if frame in self._images:
                    raise BundleValidationError("Duplicate preferred visual observation")
                self._images[frame] = record
            if record.get("image_kind") == "source_rgb":
                if self._path(record["path"]) != self._path(self.source_video_ref["path"]):
                    raise BundleValidationError("source_rgb must reference the bound source video")
                self._verify_ref(record)
            elif "path" in record:
                self._verify_ref(record)
        if set(self._images) != set(range(self.num_frames)):
            raise BundleValidationError("Preferred visual view must index every source frame")
        for frame in np.flatnonzero(self.transitions["transition_valid"]):
            if not self._images[int(frame)].get("valid", False) or not self._images[int(frame)+1].get("valid", False):
                raise BundleValidationError("A valid transition uses an invalid visual observation")

    def _require_training_ready(self) -> None:
        if self.training_readiness != "training_ready":
            raise TrainingReadinessError(f"Bundle is {self.training_readiness}: {self.readiness_reasons}")
        if self.readiness_reasons or not np.any(self.transitions["transition_valid"]):
            raise TrainingReadinessError("Training requires no blocking reasons and at least one QC-valid transition")
        if self.bundle.get("admission", {}).get("status") != "pass":
            raise TrainingReadinessError("Historical source admission has not passed")
        if self.action_profile != ACTION_PROFILE or self.hand is None or self.canonical is None or self.robot is None:
            raise TrainingReadinessError("Required real action/hand/robot profile is missing")
        if self.supervision["blocking_reason_codes"]:
            raise TrainingReadinessError(f"Supervision profile {self.supervision_profile}: {self.supervision['blocking_reason_codes']}")
        if self.supervision_profile == STRICT_STEPS:
            for i in np.flatnonzero(self.transitions["transition_valid"]):
                if not self._frame_steps[int(i)] or not self._frame_steps[int(i)+1]:
                    raise TrainingReadinessError("Strict Step profile requires original Step labels at both transition endpoints")
        elif self.supervision_profile == SOURCE_TASK:
            task = self.supervision["source_task"]
            if task["availability"] != "available" or task["source_hdf5_ref"] != self.source_hdf5_ref:
                raise TrainingReadinessError("Task-only profile requires the original bound EgoDex HDF5 instruction")
        if self.image_kind != "depth_composite_rgb":
            raise TrainingReadinessError("Training profile requires depth_composite_rgb")
        if self.visual_index.get("training_ready") is not True:
            raise TrainingReadinessError("Visual index has not satisfied its training gates")
        from .visual import _review
        for name, scope in (("m3", "episode_projection_video"), ("m5", "episode_human_arm_removal"),
                            ("m6", "episode_depth_composite")):
            gate = self.visual_index.get("gates", {}).get(name, {})
            if gate.get("status") != "accepted" or gate.get("scope") != scope or not gate.get("review_ref") or not gate.get("review_base"):
                raise TrainingReadinessError(f"{name} requires its actual scoped user review receipt")
            try:
                reviewed = _review(Path(gate["review_base"]), gate["review_ref"], self.source_video_ref["sha256"],
                                   scope, gate.get("required_evidence", {}),
                                   self.bundle["source_key"] if name != "m3" else None)
            except (OSError, ValueError, KeyError) as exc:
                raise TrainingReadinessError(f"Cannot validate {name} receipt: {exc}") from exc
            if reviewed["status"] != "accepted":
                raise TrainingReadinessError(f"{name} receipt does not accept these source/evidence hashes")
        for record in self._images.values():
            if record.get("preview_only", True) or record.get("evidence_kind") in {"mock", "synthetic_fixture", "unavailable"}:
                raise TrainingReadinessError("Mock/synthetic/preview visual cannot satisfy training gates")
            if record.get("depth_mode") not in {"metric", "aligned_metric"}:
                raise TrainingReadinessError("Training visual lacks metric depth evidence")
        for status in self.bundle.get("module_status", {}).values():
            if isinstance(status, dict) and status.get("evidence_kind") in {"mock", "synthetic_fixture"}:
                raise TrainingReadinessError("Mock/synthetic module cannot satisfy training gates")

    def __len__(self) -> int:
        return self.num_frames - 1

    def _decode_source(self, frame: int) -> np.ndarray:
        path = self._path(self.source_video_ref["path"])
        if self._video_capture is None or self._video_path != path or frame < self._video_next:
            if self._video_capture is not None:
                self._video_capture.release()
            self._video_capture = cv2.VideoCapture(str(path))
            self._video_path, self._video_next = path, 0
            if not self._video_capture.isOpened():
                raise BundleValidationError(f"Cannot decode source video: {path}")
        bgr = None
        while self._video_next <= frame:
            ok, bgr = self._video_capture.read()
            if not ok:
                raise BundleValidationError(f"S5_SOURCE_LENGTH_MISMATCH at source frame {self._video_next}")
            self._video_next += 1
        assert bgr is not None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_image(self, frame_index: int) -> np.ndarray:
        if not isinstance(frame_index, (int, np.integer)) or not 0 <= frame_index < self.num_frames:
            raise IndexError(frame_index)
        frame = int(frame_index)
        if frame in self._image_cache:
            return self._image_cache[frame].copy()
        record = self._images[frame]
        if self.image_kind == "source_rgb":
            image = self._decode_source(frame)
            expected = record.get("decoded_frame_sha256")
            if not expected or rgb_sha256(image) != expected:
                raise BundleValidationError(f"Decoded RGB SHA256 mismatch at frame {frame}")
        else:
            path = self._verify_ref(record)
            # imread's filename handling varies on Windows non-ASCII paths.
            bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise BundleValidationError(f"Unreadable image: {path}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if image.shape != (record["height"], record["width"], 3):
            raise BundleValidationError(f"Image dimensions mismatch at frame {frame}")
        # Only cache adjacent frames, so a streaming exporter has bounded RAM.
        self._image_cache = {key: val for key, val in self._image_cache.items() if key in {frame-1, frame}}
        self._image_cache[frame] = image
        return image.copy()

    def reason_codes(self, reason_bits: int) -> list[str]:
        report = self.quality_report or {}
        registry = report.get("reason_registry", report.get("reason_bits", report.get("reason_codes", {})))
        if not isinstance(registry, dict):
            return [f"reason_bits:{reason_bits}"] if reason_bits else []
        result = []
        for key, value in registry.items():
            if isinstance(value, dict):
                bit = value.get("mask", value.get("value"))
                if bit is None and isinstance(value.get("bit"), int):
                    bit = 1 << value["bit"]
                name = value.get("code", key)
                if isinstance(bit, int) and reason_bits & bit:
                    result.append(str(name))
            elif isinstance(value, int):
                if reason_bits & value:
                    result.append(str(key))
            else:
                try:
                    bit = int(key)
                except (ValueError, TypeError):
                    continue
                if reason_bits & bit:
                    result.append(str(value))
        return result or ([f"reason_bits:{reason_bits}"] if reason_bits else [])

    def frame_metadata(self, frame: int) -> dict[str, Any]:
        if not 0 <= frame < self.num_frames:
            raise IndexError(frame)
        qc = {key: _jsonable(value[frame]) for key, value in self.quality.items() if value.ndim and value.shape[0] == self.num_frames}
        return {
            "frame_index": frame, "timestamp_s": float(self.timestamp_s[frame]),
            "image_kind": self.image_kind, "visual": self._images[frame],
            "step_meta": [self._steps_by_id[sid] for sid in self._frame_steps[frame]],
            "step_sidecar_status": self.supervision["step_sidecar_status"],
            "supervision_profile": self.supervision_profile,
            "quality": qc, "reason_codes": self.reason_codes(int(self.quality["reason_bits"][frame])),
        }

    def get_transition(self, i: int) -> dict[str, Any]:
        if not isinstance(i, (int, np.integer)) or not 0 <= i < len(self):
            raise IndexError(i)
        i = int(i)
        valid = bool(self.transitions["transition_valid"][i])
        if self.mode == "train" and not valid:
            raise TrainingReadinessError(f"Invalid transition {i}: {self.reason_codes(int(self.transitions['reason_bits'][i]))}")
        source_meta, target_meta = self.frame_metadata(i), self.frame_metadata(i+1)
        result = {
            "episode_id": self.episode_id, "source_frame_index": i, "target_frame_index": i+1,
            "timestamp_s": float(self.timestamp_s[i]), "next_timestamp_s": float(self.timestamp_s[i+1]),
            "dt_s": float(self.transitions["dt_s"][i]),
            "image_rgb": self.get_image(i), "next_image_rgb": self.get_image(i+1),
            "action7": self.transitions["action7"][i].copy(), "action_profile": self.action_profile,
            "robot_qpos": None if self.robot is None else self.robot["qpos"][i].copy(),
            "next_robot_qpos": None if self.robot is None else self.robot["qpos"][i+1].copy(),
            "robot_state": None if self.robot is None else {key: value[i].copy() for key, value in self.robot.items() if value.ndim and value.shape[0] == self.num_frames},
            "next_robot_state": None if self.robot is None else {key: value[i+1].copy() for key, value in self.robot.items() if value.ndim and value.shape[0] == self.num_frames},
            "canonical_abs_ref": self.bundle.get("refs", {}).get("canonical"),
            "task_meta": self.task_meta, "source_metadata": self.source_metadata,
            "supervision_profile": self.supervision_profile, "supervision": self.supervision,
            "step_sidecar_status": self.supervision["step_sidecar_status"],
            "step_meta": source_meta["step_meta"] or None,
            "next_step_meta": target_meta["step_meta"] or None,
            "crosses_step_boundary": self._frame_steps[i] != self._frame_steps[i+1],
            "object_association_status": self.object_associations.get("associations", []),
            "quality": source_meta["quality"], "next_quality": target_meta["quality"],
            "transition_valid": valid,
            "reason_bits": int(self.transitions["reason_bits"][i]),
            "reason_codes": self.reason_codes(int(self.transitions["reason_bits"][i])),
            "training_readiness": self.training_readiness, "readiness_reasons": self.readiness_reasons,
            "observation_kind": self.image_kind, "provenance_refs": self.bundle.get("refs", {}),
            "provenance": self.bundle.get("provenance", {}),
        }
        return result

    def iter_transitions(self, valid_only: bool = False) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            if valid_only and not bool(self.transitions["transition_valid"][i]):
                continue
            yield self.get_transition(i)

    def close(self) -> None:
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        self._image_cache.clear()

    def __enter__(self) -> "BundleReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def open_bundle(path: str | Path, mode: str = "train") -> BundleReader:
    return BundleReader(path, mode=mode)
