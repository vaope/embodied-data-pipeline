"""Versioned data contracts (system-design.md section 6).

Two families of type live here:

* **Document schemas** (:class:`EpisodeManifest`, :class:`CameraModel`,
  :class:`QualityReport`, :class:`StageManifest`) -- Pydantic models that are
  written as JSON/YAML and validated field-by-field.
* **Array schemas** (:class:`HandSequence`, :class:`CanonicalActionSequence`,
  :class:`RobotTrajectory`, :class:`RenderBundle`) -- numpy-backed records
  persisted as ``.npz``.  Their shape, dtype, frame and unit are declared in
  :data:`ARRAY_SPECS` and enforced by :meth:`ArrayRecord.validate`, because
  system-design.md 6 forbids relying on comments for that.

Every schema carries ``schema_version``.  A semantic change means a new
version string, never an in-place redefinition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .frames import Frame, TransformName, parse_transform_name

__all__ = [
    "ARRAY_SPECS",
    "ArraySpec",
    "CameraModel",
    "CameraFile",
    "CalibrationSource",
    "DepthMode",
    "CanonicalActionSequence",
    "EpisodeManifest",
    "EpisodeSource",
    "HAND_SIDE_SIGN",
    "HandSequence",
    "Provisionance",
    "QualityReport",
    "RenderBundle",
    "RobotConfig",
    "RobotTrajectory",
    "StageManifest",
    "TaskSpec",
    "Timebase",
    "ValidationIssue",
    "ValidationResult",
    "array_spec",
    "describe_pose_variable",
]

#: ``handedness`` sign per hand side ([PAPER] appendix A.3, ADR-0021).
HAND_SIDE_SIGN: dict[str, int] = {"right": 1, "left": -1}

# ---------------------------------------------------------------------------
# Array specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArraySpec:
    """Declared shape/dtype/frame/unit semantics for one persisted array."""

    name: str
    trailing_shape: tuple[int | str, ...]
    dtype: str
    frame: str
    unit: str
    description: str
    optional: bool = False

    def describe(self) -> str:
        shape = ", ".join(str(s) for s in self.trailing_shape)
        opt = " (optional)" if self.optional else ""
        hand = "H is the hand axis" if self.trailing_shape and self.trailing_shape[0] == "K" else ""
        return (
            f"{self.name}: [T, H, {shape}] {self.dtype} frame={self.frame} "
            f"unit={self.unit} -- {self.description}{opt}"
            + (f" ({hand})" if hand else "")
        )


def _spec(
    name: str,
    trailing_shape: tuple[int | str, ...],
    dtype: str,
    frame: str,
    unit: str,
    description: str,
    optional: bool = False,
) -> ArraySpec:
    return ArraySpec(name, trailing_shape, dtype, frame, unit, description, optional)


ARRAY_SPECS: dict[str, dict[str, ArraySpec]] = {
    "hand": {
        "timestamp_s": _spec("timestamp_s", (), "float64", "n/a", "s", "PTS-derived time"),
        "frame_index": _spec("frame_index", (), "int64", "n/a", "1", "index into the decoded video"),
        "keypoints_C_m": _spec(
            "keypoints_C_m", (21, 3), "float32", "C", "m", "per-joint 3D position in camera optical frame"
        ),
        "keypoints_uv_px": _spec(
            "keypoints_uv_px", (21, 2), "float32", "C", "px", "pinhole projection of keypoints_C_m"
        ),
        "handedness": _spec("handedness", (), "int8", "n/a", "1", "-1 left, +1 right, 0 unknown"),
        "detection_confidence": _spec("detection_confidence", (), "float32", "n/a", "1", "per-hand score"),
        "valid": _spec("valid", (), "bool", "n/a", "1", "per-hand validity flag"),
        "keypoint_confidence": _spec(
            "keypoint_confidence", ("K",), "float32", "n/a", "1", "per-joint confidence", optional=True
        ),
    },
    "canonical": {
        "timestamp_s": _spec("timestamp_s", (), "float64", "n/a", "s", "PTS-derived time"),
        "frame_index": _spec("frame_index", (), "int64", "n/a", "1", "index into the decoded video"),
        "T_C_TCP": _spec("T_C_TCP", (4, 4), "float64", "C<-TCP", "m, rad", "absolute TCP pose"),
        "tcp_position_C_m": _spec("tcp_position_C_m", (3,), "float64", "C", "m", "TCP translation"),
        "tcp_quaternion_C_xyzw": _spec(
            "tcp_quaternion_C_xyzw", (4,), "float64", "C<-TCP", "1", "TCP orientation, xyzw"
        ),
        "gripper_width_m": _spec("gripper_width_m", (), "float64", "n/a", "m", "jaw opening"),
        "confidence": _spec("confidence", (), "float32", "n/a", "1", "retarget confidence"),
        "valid": _spec("valid", (), "bool", "n/a", "1", "retarget validity"),
        "fallback_code": _spec("fallback_code", (), "uint16", "n/a", "1", "reason code registry"),
        "so3_correction_rad": _spec(
            "so3_correction_rad", (), "float32", "n/a", "rad", "SVD repair magnitude", optional=True
        ),
    },
    "canonical_delta": {
        "delta_position_C_m": _spec(
            "delta_position_C_m", (3,), "float32", "C", "m", "p[t+1] - p[t]"
        ),
        "delta_rotvec_C_rad": _spec(
            "delta_rotvec_C_rad", (3,), "float32", "C", "rad", "Log_SO3(R[t+1] R[t]^T)"
        ),
        "gripper_command": _spec(
            "gripper_command", (1,), "float32", "n/a", "1", "normalised gripper command (export profile)"
        ),
    },
    "robot": {
        "timestamp_s": _spec("timestamp_s", (), "float64", "n/a", "s", "PTS-derived time"),
        "frame_index": _spec("frame_index", (), "int64", "n/a", "1", "index into the decoded video"),
        "qpos": _spec("qpos", ("dof",), "float64", "n/a", "rad/m", "arm joint positions"),
        "qvel": _spec("qvel", ("dof",), "float64", "n/a", "rad/s", "arm joint velocities"),
        "gripper_qpos": _spec("gripper_qpos", ("gripper_dof",), "float64", "n/a", "m", "finger joint positions"),
        "T_B_TCP_target": _spec("T_B_TCP_target", (4, 4), "float64", "B<-TCP", "m, rad", "IK target"),
        "T_B_TCP_fk": _spec("T_B_TCP_fk", (4, 4), "float64", "B<-TCP", "m, rad", "FK of solved qpos"),
        "position_error_m": _spec("position_error_m", (), "float64", "n/a", "m", "FK vs target position error"),
        "rotation_error_rad": _spec("rotation_error_rad", (), "float64", "n/a", "rad", "FK vs target rotation error"),
        "ik_success": _spec("ik_success", (), "bool", "n/a", "1", "solver reported success"),
        "collision_code": _spec("collision_code", (), "uint16", "n/a", "1", "contact classification"),
        "joint_limit_hit": _spec("joint_limit_hit", (), "bool", "n/a", "1", "joint limit violation"),
        "ik_iterations": _spec("ik_iterations", (), "int32", "n/a", "1", "solver iterations used"),
        "solver_status": _spec("solver_status", (), "uint16", "n/a", "1", "IK status code"),
    },
    "render": {
        "frame_index": _spec("frame_index", (), "int64", "n/a", "1", "index into the decoded video"),
        "robot_rgb": _spec("robot_rgb", ("H_img", "W_img", 3), "uint8", "n/a", "1", "RGB robot-only render", True),
        "robot_mask": _spec("robot_mask", ("H_img", "W_img"), "bool", "n/a", "1", "arm body mask", True),
        "gripper_mask": _spec("gripper_mask", ("H_img", "W_img"), "bool", "n/a", "1", "gripper mask", True),
        "robot_depth_m": _spec(
            "robot_depth_m", ("H_img", "W_img"), "float32", "C", "m", "metric depth, NaN where invalid", True
        ),
    },
}

def _match_shape(
    actual: tuple[int, ...], declared: tuple[int | str, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Align ``actual`` against ``declared``, resolving named axes positionally.

    Named axes (e.g. ``"K"`` for the 21 MANO keypoints, ``"dof"`` for the joint
    count) are compared by *count* rather than by a hard-coded size, because the
    specification explicitly allows those dimensions to be discovered from the
    data.  Literal sizes must agree exactly.
    """
    resolved = tuple(
        (int(d) if isinstance(d, int) else actual[i] if i < len(actual) else -1)
        for i, d in enumerate(declared)
    )
    return actual, resolved


def array_spec(table: str, name: str) -> ArraySpec:
    """Fetch the declared spec for ``table.name`` or raise a clear error."""
    if table not in ARRAY_SPECS:
        raise KeyError(f"unknown array table {table!r}; known: {sorted(ARRAY_SPECS)}")
    spec = ARRAY_SPECS[table].get(name)
    if spec is None:
        raise KeyError(
            f"array {name!r} is not declared in table {table!r}; "
            f"declared: {sorted(ARRAY_SPECS[table])}"
        )
    return spec


def describe_pose_variable(variable_name: str) -> str:
    """Answer "which frame to which frame, and in what unit" for a pose variable.

    system-design.md Gate M0 requires that any persisted pose can answer this.
    """
    for table, specs in ARRAY_SPECS.items():
        spec = specs.get(variable_name)
        if spec is not None:
            if spec.frame in {"n/a", "1"} or "<-" not in spec.frame:
                return f"{table}.{variable_name}: frame={spec.frame}, unit={spec.unit}"
            source, target = spec.frame.split("<-")
            name = TransformName(source=Frame(source), target=Frame(target))
            return f"{table}.{variable_name}: {name.describe()}, unit={spec.unit}"
    if variable_name.startswith("T_"):
        name = parse_transform_name(variable_name)
        return name.describe()
    raise KeyError(f"no declared frame/unit for variable {variable_name!r}")


# ---------------------------------------------------------------------------
# Validation result plumbing
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single validation problem, with enough context to act on it."""

    code: str
    message: str
    severity: str = "error"
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - presentation
        return f"[{self.severity}] {self.code}: {self.message}"


@dataclass
class ValidationResult:
    """Aggregated validation outcome."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def add(self, code: str, message: str, severity: str = "error", **context: Any) -> None:
        self.issues.append(ValidationIssue(code, message, severity, context))

    def extend(self, other: ValidationResult) -> None:
        self.issues.extend(other.issues)

    def raise_if_bad(self) -> None:
        if not self.ok:
            detail = "; ".join(str(i) for i in self.errors)
            raise ValueError(f"schema validation failed: {detail}")

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "ok"
        parts = [f"{len(self.errors)} error(s)", f"{len(self.warnings)} warning(s)"]
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Array record base
# ---------------------------------------------------------------------------


class ArrayRecord:
    """Base class for numpy-backed, ``.npz``-persisted schemas.

    Subclasses declare ``schema_version``, ``table`` (the key into
    :data:`ARRAY_SPECS`), the leading hand dimension policy and which arrays
    are required.
    """

    schema_version: ClassVar[str] = ""
    table: ClassVar[str] = ""
    #: names of arrays that must be present
    required: ClassVar[tuple[str, ...]] = ()
    #: arrays whose *second* axis is the hand dimension (``[T, H, ...]``)
    hand_axis_names: ClassVar[tuple[str, ...]] = ()
    #: arrays indexed by hand **and** by keypoint (``[T, H, K, ...]``)
    hand_and_joint_axis_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, arrays: Mapping[str, Any], *, meta: Mapping[str, Any] | None = None) -> None:
        self.arrays: dict[str, NDArray[Any]] = {
            k: np.asarray(v) for k, v in arrays.items()
        }
        self.meta: dict[str, Any] = dict(meta or {})

    # -- construction -------------------------------------------------------

    @classmethod
    def from_arrays(cls, **arrays: Any) -> ArrayRecord:
        return cls(arrays)

    # -- dunder -------------------------------------------------------------

    def __getitem__(self, name: str) -> NDArray[Any]:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(
                f"{type(self).__name__} has no array {name!r}; present: {sorted(self.arrays)}"
            ) from exc

    def __contains__(self, name: str) -> bool:
        return name in self.arrays

    def __len__(self) -> int:
        for name in ("timestamp_s", "frame_index"):
            if name in self.arrays:
                return int(self.arrays[name].shape[0])
        return 0

    def get(self, name: str, default: Any = None) -> Any:
        return self.arrays.get(name, default)

    @property
    def num_frames(self) -> int:
        return len(self)

    # -- validation ---------------------------------------------------------

    def validate(self, expect_hands: int | None = None) -> ValidationResult:
        """Check declared shape/dtype/frame/unit semantics for every array."""
        res = ValidationResult()
        if not self.schema_version:
            res.add("schema.missing_version", f"{type(self).__name__} has no schema_version")
        for name in self.required:
            if name not in self.arrays:
                res.add(
                    "schema.missing_array",
                    f"{type(self).__name__} is missing required array {name!r}",
                    array=name,
                )
        if not self.arrays:
            res.add("schema.empty", f"{type(self).__name__} carries no arrays")
            return res

        n_frames = len(self)
        for name, arr in self.arrays.items():
            try:
                spec = array_spec(self.table, name)
            except KeyError as exc:
                res.add("schema.undeclared_array", str(exc), array=name)
                continue
            if not np.issubdtype(arr.dtype, np.dtype(spec.dtype)):
                res.add(
                    "schema.dtype_mismatch",
                    f"{name}: declared {spec.dtype}, got {arr.dtype}",
                    array=name,
                )
            if arr.ndim == 0:
                if n_frames != 1:
                    res.add(
                        "schema.shape_mismatch",
                        f"{name}: scalar array with {n_frames} frames",
                        array=name,
                    )
                continue
            if arr.shape[0] != n_frames:
                res.add(
                    "schema.shape_mismatch",
                    f"{name}: leading dim {arr.shape[0]} != T={n_frames}",
                    array=name,
                )
                continue
            expected = tuple(
                (d if isinstance(d, int) else arr.shape[-len(spec.trailing_shape) + i])
                for i, d in enumerate(spec.trailing_shape)
            )
            if name in self.hand_axis_names or name in self.hand_and_joint_axis_names:
                if arr.ndim < 2:
                    res.add(
                        "schema.shape_mismatch",
                        f"{name}: hand-indexed array must have shape [T, H, ...], got {arr.shape}",
                        array=name,
                    )
                    continue
                if expect_hands is not None and arr.shape[1] != expect_hands:
                    res.add(
                        "schema.shape_mismatch",
                        f"{name}: H={arr.shape[1]} != expected {expect_hands}",
                        array=name,
                    )
                if name in self.hand_and_joint_axis_names:
                    if arr.ndim < 3:
                        res.add(
                            "schema.shape_mismatch",
                            f"{name}: expected [T, H, K, ...] with a joint axis, got {arr.shape}",
                            array=name,
                        )
                        continue
                    # the declared shape starts with the joint axis, which the
                    # [T, H] prefix did not consume
                    got, expected = _match_shape(arr.shape[2:], spec.trailing_shape)
                else:
                    got, expected = _match_shape(arr.shape[2:], spec.trailing_shape)
            else:
                got, expected = _match_shape(arr.shape[1:], spec.trailing_shape)
            if got != expected:
                res.add(
                    "schema.shape_mismatch",
                    f"{name}: trailing shape {got} != declared {expected}",
                    array=name,
                )
        return res

    def validate_numeric(self, *, allow_nan_in: Iterable[str] = ()) -> ValidationResult:
        """Check for unexpected NaN/Inf outside the explicitly allowed arrays."""
        res = ValidationResult()
        allowed = set(allow_nan_in)
        for name, arr in self.arrays.items():
            if name in allowed or arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.floating):
                continue
            if arr.size and not np.all(np.isfinite(arr)):
                n_bad = int(np.count_nonzero(~np.isfinite(arr)))
                res.add(
                    "schema.non_finite",
                    f"{name}: {n_bad} non-finite value(s)",
                    array=name,
                )
        return res

    def assert_valid(self, expect_hands: int | None = None) -> None:
        self.validate(expect_hands=expect_hands).raise_if_bad()

    # -- hand identity (ADR-0021) ------------------------------------------

    @property
    def num_hands_independent(self) -> int:
        """``H`` read from the hand-indexed arrays, without trusting ``meta``."""
        shapes = {
            int(self.arrays[name].shape[1])
            for name in self.hand_axis_names + self.hand_and_joint_axis_names
            if name in self.arrays and self.arrays[name].ndim >= 2
        }
        return shapes.pop() if len(shapes) == 1 else 0

    @property
    def hand_ids(self) -> list[str]:
        """Stable hand identity in ``H`` order, from the persisted metadata."""
        ids = self.meta.get("hand_ids")
        return [str(v) for v in ids] if isinstance(ids, (list, tuple)) else []

    def validate_hand_identity(self) -> ValidationResult:
        """Check the ``[T, H, ...]`` hand axis and its persisted identity.

        ADR-0021 requires two things beyond the shape check in
        :meth:`validate`:

        1. every hand-indexed array agrees on ``H`` -- a per-array mismatch would
           otherwise silently truncate at the first consumer;
        2. a multi-hand sequence carries ``hand_ids``, and those ids agree with
           the ``handedness`` sign.  A single-hand sequence without ids stays
           readable (old ``hand.v1`` files), but the recovered identity is
           reported as a warning instead of being assumed.
        """
        res = ValidationResult()
        shapes = {
            name: int(self.arrays[name].shape[1])
            for name in self.hand_axis_names + self.hand_and_joint_axis_names
            if name in self.arrays and self.arrays[name].ndim >= 2
        }
        distinct = sorted(set(shapes.values()))
        if len(distinct) > 1:
            res.add(
                "schema.hand_axis_mismatch",
                f"hand-indexed arrays disagree on H: {{"
                + ", ".join(f"{k}: {v}" for k, v in sorted(shapes.items()))
                + "}",
            )
            return res
        hands = distinct[0] if distinct else 0
        if hands == 0:
            res.add("schema.no_hand_axis", f"{type(self).__name__} carries no hand-indexed array")
            return res

        ids = self.hand_ids
        if ids and len(ids) != hands:
            res.add(
                "schema.hand_ids_length",
                f"meta.hand_ids has {len(ids)} entries but H={hands}",
            )
        if ids and len(set(ids)) != len(ids):
            res.add("schema.hand_ids_duplicate", f"meta.hand_ids repeats an id: {ids}")
        bad_ids = [i for i in ids if i not in HAND_SIDE_SIGN]
        if bad_ids:
            res.add(
                "schema.hand_ids_unknown",
                f"meta.hand_ids contains unknown side(s) {bad_ids}; use 'left'/'right'",
            )
        if hands > 1 and not ids:
            res.add(
                "schema.hand_ids_missing",
                f"H={hands} requires meta.hand_ids so a consumer cannot infer identity "
                "from array position (ADR-0021)",
            )
        if hands == 1 and not ids:
            res.add(
                "schema.hand_ids_recovered",
                "no meta.hand_ids; the single hand's identity is recovered from "
                "handedness/provenance for legacy hand.v1 files",
                severity="warning",
            )

        if "handedness" in self.arrays and ids:
            handedness = np.asarray(self.arrays["handedness"])
            for h, hand_id in enumerate(ids):
                if hand_id not in HAND_SIDE_SIGN:
                    continue
                column = handedness[:, h]
                finite = column[column != 0]
                if finite.size == 0:
                    res.add(
                        "schema.handedness_unknown",
                        f"hand {h} ({hand_id}) has no signed handedness value",
                    )
                    continue
                wrong = int(np.count_nonzero(finite != HAND_SIDE_SIGN[hand_id]))
                if wrong:
                    res.add(
                        "schema.handedness_identity_conflict",
                        f"hand {h} is declared {hand_id!r} but {wrong} frame(s) carry "
                        f"handedness {sorted(set(int(v) for v in finite))}",
                    )
        return res

    # -- io -----------------------------------------------------------------

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: np.asarray(v) for k, v in self.arrays.items()}
        payload["__schema_version__"] = np.asarray(self.schema_version)
        if self.meta:
            import json

            payload["__meta_json__"] = np.asarray(json.dumps(self.meta, sort_keys=True, default=str))
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> ArrayRecord:
        import json

        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            arrays: dict[str, NDArray[Any]] = {}
            meta: dict[str, Any] = {}
            version = ""
            for key in data.files:
                if key == "__schema_version__":
                    version = str(data[key].item())
                elif key == "__meta_json__":
                    meta = json.loads(str(data[key].item()))
                else:
                    arrays[key] = data[key]
        if version and cls.schema_version and version != cls.schema_version:
            raise ValueError(
                f"{path}: schema version mismatch, file has {version!r}, "
                f"reader expects {cls.schema_version!r}"
            )
        return cls(arrays, meta=meta)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "num_frames": len(self),
            "num_hands": self.num_hands_independent,
            "hand_ids": self.hand_ids,
            "arrays": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in self.arrays.items()
            },
            "meta": self.meta,
        }


class HandSequence(ArrayRecord):
    """Canonical hand perception output (system-design.md 6.2)."""

    schema_version: ClassVar[str] = "ego2robot.hand.v1"
    table: ClassVar[str] = "hand"
    required: ClassVar[tuple[str, ...]] = (
        "timestamp_s",
        "frame_index",
        "keypoints_C_m",
        "keypoints_uv_px",
        "handedness",
        "detection_confidence",
        "valid",
    )
    hand_axis_names: ClassVar[tuple[str, ...]] = (
        "keypoints_C_m",
        "keypoints_uv_px",
        "handedness",
        "detection_confidence",
        "valid",
    )
    hand_and_joint_axis_names: ClassVar[tuple[str, ...]] = ("keypoint_confidence",)

    @property
    def num_hands(self) -> int:
        return int(self.arrays["valid"].shape[1]) if "valid" in self.arrays else 0


class CanonicalActionSequence(ArrayRecord):
    """Robot-independent canonical gripper trajectory (system-design.md 6.3)."""

    schema_version: ClassVar[str] = "ego2robot.canonical_action.v1"
    table: ClassVar[str] = "canonical"
    required: ClassVar[tuple[str, ...]] = (
        "timestamp_s",
        "frame_index",
        "T_C_TCP",
        "tcp_position_C_m",
        "tcp_quaternion_C_xyzw",
        "gripper_width_m",
        "confidence",
        "valid",
        "fallback_code",
    )
    hand_axis_names: ClassVar[tuple[str, ...]] = (
        "T_C_TCP",
        "tcp_position_C_m",
        "tcp_quaternion_C_xyzw",
        "gripper_width_m",
        "confidence",
        "valid",
        "fallback_code",
        "so3_correction_rad",
    )

    @property
    def num_hands(self) -> int:
        return int(self.arrays["valid"].shape[1]) if "valid" in self.arrays else 0

    @property
    def delta_table(self) -> str:
        return "canonical_delta"

    def relative_action(
        self, gripper_profile: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
    ) -> dict[str, NDArray[Any]]:
        """Derive the camera-frame relative 7D action without touching absolutes.

        system-design.md 6.3: the absolute ``T_C_TCP`` stays authoritative; the
        delta arrays are derived views that an exporter consumes.  Deltas are
        computed per hand, so the returned arrays keep the ``[T-1, H, ...]``
        shape the schema declares.
        """
        from .transforms import delta_position, delta_rotvec

        pos = self.arrays["tcp_position_C_m"]  # [T, H, 3]
        R = self.arrays["T_C_TCP"][:, :, :3, :3]  # [T, H, 3, 3]
        width = self.arrays["gripper_width_m"]  # [T, H]
        n_hands = pos.shape[1]

        delta_p = np.stack(
            [delta_position(pos[:, h, :]) for h in range(n_hands)], axis=1
        )
        delta_w = np.stack(
            [delta_rotvec(R[:, h, :, :]) for h in range(n_hands)], axis=1
        )
        out: dict[str, NDArray[Any]] = {
            "delta_position_C_m": delta_p.astype(np.float32),
            "delta_rotvec_C_rad": delta_w.astype(np.float32),
        }
        if gripper_profile is None:
            out["gripper_command"] = width[:-1, :, None].astype(np.float32)
        else:
            out["gripper_command"] = gripper_profile(width).astype(np.float32)[:-1, :, None]
        return out


class RobotTrajectory(ArrayRecord):
    """Per-frame IK/FK result in the robot base frame (system-design.md 6.4)."""

    schema_version: ClassVar[str] = "ego2robot.robot_trajectory.v1"
    table: ClassVar[str] = "robot"
    required: ClassVar[tuple[str, ...]] = (
        "timestamp_s",
        "frame_index",
        "qpos",
        "qvel",
        "gripper_qpos",
        "T_B_TCP_target",
        "T_B_TCP_fk",
        "position_error_m",
        "rotation_error_rad",
        "ik_success",
        "collision_code",
    )
    hand_axis_names: ClassVar[tuple[str, ...]] = (
        "T_B_TCP_target",
        "T_B_TCP_fk",
        "position_error_m",
        "rotation_error_rad",
        "ik_success",
    )

    @property
    def num_hands(self) -> int:
        return int(self.arrays["ik_success"].shape[1]) if "ik_success" in self.arrays else 0


class RenderBundle(ArrayRecord):
    """MuJoCo render outputs (system-design.md 6.5)."""

    schema_version: ClassVar[str] = "ego2robot.render_bundle.v1"
    table: ClassVar[str] = "render"
    required: ClassVar[tuple[str, ...]] = ("frame_index",)
    hand_axis_names: ClassVar[tuple[str, ...]] = ()


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------

SCHEMA_CONFIG = ConfigDict(extra="forbid", validate_assignment=True)


class CalibrationSource(str):
    """Enum-like string for how camera parameters were obtained."""

    CALIBRATED = "calibrated"
    DATASET = "dataset"
    ESTIMATED = "estimated"
    ASSUMED = "assumed"

    ALL = ("calibrated", "dataset", "estimated", "assumed")


class DepthMode(str):
    """Depth metricity label (system-design.md 9.5)."""

    METRIC = "metric"
    ALIGNED = "aligned"
    RELATIVE = "relative"

    ALL = ("metric", "aligned", "relative")


class CameraModel(BaseModel):
    """Pinhole camera model with an explicit provenance label."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, populate_by_name=True
    )

    model_type: str = Field(default="pinhole", alias="model")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    K: list[list[float]]
    distortion: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    calibration_source: str = "assumed"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    #: set to True only when the source is calibrated/dataset; 'assumed' is debug-only
    debug_only: bool = False

    @field_validator("K")
    @classmethod
    def _check_K(cls, v: list[list[float]]) -> list[list[float]]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got shape {arr.shape}")
        if arr[2, 2] != 1.0:
            raise ValueError("K[2,2] must be 1.0")
        if arr[0, 0] <= 0.0 or arr[1, 1] <= 0.0:
            raise ValueError("focal lengths fx, fy must be positive")
        return v

    @field_validator("distortion")
    @classmethod
    def _check_distortion(cls, v: list[float]) -> list[float]:
        if len(v) not in (0, 4, 5, 8, 12, 14):
            raise ValueError(f"unsupported distortion length {len(v)}")
        return v

    @field_validator("calibration_source")
    @classmethod
    def _check_source(cls, v: str) -> str:
        if v not in CalibrationSource.ALL:
            raise ValueError(f"calibration_source must be one of {CalibrationSource.ALL}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _check_debug_only(self) -> CameraModel:
        if self.calibration_source == CalibrationSource.ASSUMED and not self.debug_only:
            raise ValueError(
                "calibration_source='assumed' is debug-only (system-design.md 9.1); "
                "set debug_only=True so it cannot silently enter a formal dataset"
            )
        return self

    @property
    def K_array(self) -> NDArray[np.float64]:
        return np.asarray(self.K, dtype=np.float64)

    @property
    def fx(self) -> float:
        return float(self.K[0][0])

    @property
    def fy(self) -> float:
        return float(self.K[1][1])

    @property
    def cx(self) -> float:
        return float(self.K[0][2])

    @property
    def cy(self) -> float:
        return float(self.K[1][2])

    def project(self, points_C_m: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Pinhole projection of camera-frame points into pixels (no distortion)."""
        P = np.asarray(points_C_m, dtype=np.float64)
        if P.shape[-1] != 3:
            raise ValueError(f"points must be [..., 3], got {P.shape}")
        z = P[..., 2]
        safe_z = np.where(np.abs(z) < 1e-9, np.nan, z)
        u = self.fx * P[..., 0] / safe_z + self.cx
        v = self.fy * P[..., 1] / safe_z + self.cy
        return np.stack([u, v], axis=-1)

    def unproject(self, uv_px: NDArray[np.floating[Any]], depth_m: float | NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Back-project pixels at a given depth into the camera optical frame."""
        uv = np.asarray(uv_px, dtype=np.float64)
        d = np.asarray(depth_m, dtype=np.float64)
        x = (uv[..., 0] - self.cx) * d / self.fx
        y = (uv[..., 1] - self.cy) * d / self.fy
        z = np.broadcast_to(d, x.shape)
        return np.stack([x, y, z], axis=-1)


class Province(BaseModel):
    """Provenance block (kept small; see system-design.md 6.1)."""

    model_config = SCHEMA_CONFIG

    config_hash: str = ""
    code_revision: str = ""
    created_at: str = ""


# ``Provisionance`` is the spelling used in the export list above; expose both.
Provisionance = Province


class CameraFile(BaseModel):
    """Standalone ``source/camera.json`` envelope (system-design.md 9.1).

    Wraps :class:`CameraModel` with the episode-independent provenance a
    reviewer needs to answer "where did these numbers come from".
    """

    model_config = SCHEMA_CONFIG

    schema_version: str = "ego2robot.camera.v1"
    camera: CameraModel
    camera_frame: str = "C"
    provenance: str = ""

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if v != "ego2robot.camera.v1":
            raise ValueError(f"unsupported camera schema_version {v!r}")
        return v

    @field_validator("camera_frame")
    @classmethod
    def _check_frame(cls, v: str) -> str:
        if v != "C":
            raise ValueError(
                "camera_frame must be 'C': K is only valid in the camera optical "
                "frame (system-design.md 5.1)"
            )
        return v


class EpisodeSource(BaseModel):
    model_config = SCHEMA_CONFIG

    video_uri: str
    sha256: str = ""
    source_dataset: str = "local"
    hdf5_uri: str | None = None
    hdf5_sha256: str | None = None
    license: str = "unknown"
    #: never publish content whose license is unknown (system-design.md 16)
    publishable: bool = False


class Timebase(BaseModel):
    model_config = SCHEMA_CONFIG

    num_frames: int = Field(ge=0)
    nominal_fps: float = Field(gt=0.0)
    #: PTS of every frame in seconds, if the container supplied them
    timestamps_s: list[float] | None = None
    timebase_source: str = "pts"

    @model_validator(mode="after")
    def _check_timestamps(self) -> Timebase:
        if self.timestamps_s is not None:
            if len(self.timestamps_s) != self.num_frames:
                raise ValueError(
                    f"timestamps_s length {len(self.timestamps_s)} != num_frames {self.num_frames}"
                )
            arr = np.asarray(self.timestamps_s, dtype=np.float64)
            if arr.size and not np.all(np.diff(arr) > 0.0):
                raise ValueError("timestamps_s must be strictly increasing")
        return self

    @property
    def duration_s(self) -> float:
        if self.timestamps_s:
            return float(self.timestamps_s[-1] - self.timestamps_s[0])
        return 0.0


class TaskSpec(BaseModel):
    model_config = SCHEMA_CONFIG

    instruction: str = ""
    subtask: str | None = None


class EpisodeManifest(BaseModel):
    """``ego2robot.episode.v1`` (system-design.md 6.1)."""

    model_config = SCHEMA_CONFIG

    schema_version: str = "ego2robot.episode.v1"
    episode_id: str
    source: EpisodeSource
    timebase: Timebase
    camera: CameraModel
    task: TaskSpec = Field(default_factory=TaskSpec)
    provenance: Province = Field(default_factory=Province)
    #: T_C_B used for this episode, when the profile pins a base placement
    T_C_B: list[list[float]] | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if v != "ego2robot.episode.v1":
            raise ValueError(f"unsupported episode schema_version {v!r}")
        return v

    @model_validator(mode="after")
    def _check_T_C_B(self) -> EpisodeManifest:
        if self.T_C_B is not None:
            arr = np.asarray(self.T_C_B, dtype=np.float64)
            if arr.shape != (4, 4):
                raise ValueError(f"T_C_B must be 4x4, got shape {arr.shape}")
            if not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
                raise ValueError("T_C_B bottom row must be [0, 0, 0, 1]")
            R = arr[:3, :3]
            ortho = float(np.max(np.abs(R.T @ R - np.eye(3))))
            det = float(np.linalg.det(R))
            if ortho > 1e-6:
                raise ValueError(
                    f"T_C_B rotation block is not orthonormal (max |R^T R - I| = {ortho:.2e}); "
                    "a rotation matrix is required, not an arbitrary 4x4"
                )
            if abs(det - 1.0) > 1e-6:
                raise ValueError(
                    f"T_C_B rotation block has determinant {det:+.6f}; det = -1 is a "
                    "reflection and cannot map a right-handed frame onto a right-handed frame"
                )
        return self

    @property
    def T_C_B_array(self) -> NDArray[np.float64] | None:
        return None if self.T_C_B is None else np.asarray(self.T_C_B, dtype=np.float64)

    def transform_to_base(self, T_C_TCP: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """``T_B_TCP = inverse(T_C_B) @ T_C_TCP`` (system-design.md 5.2)."""
        from .transforms import invert_transform

        if self.T_C_B is None:
            raise ValueError(
                "episode has no T_C_B; a robot base placement is required before robot binding"
            )
        return invert_transform(self.T_C_B_array) @ np.asarray(T_C_TCP, dtype=np.float64)


class RobotConfig(BaseModel):
    """Robot description + placement contract (system-design.md 6.4 / 8.1)."""

    model_config = SCHEMA_CONFIG

    robot_id: str
    dof: int = Field(gt=0)
    gripper_dof: int = Field(ge=0)
    max_reach_m: float = Field(gt=0.0)
    home_qpos: list[float]
    home_gripper_qpos: list[float] = Field(default_factory=list)
    mjcf_uri: str
    mjcf_sha256: str = ""
    mjcf_license: str = "unknown"
    mjcf_source: str = ""
    #: T_E_TCP: constant offset from the MJCF end-effector body to the grasp frame
    T_E_TCP: list[list[float]]
    joint_names: list[str]
    gripper_joint_names: list[str] = Field(default_factory=list)
    actuator_names: list[str] = Field(default_factory=list)
    position_limits: list[list[float]] = Field(default_factory=list)
    eef_body_name: str
    base_body_name: str = "link0"
    gripper_geom_names: list[str] = Field(default_factory=list)
    #: maximum allowed per-frame joint jump before QC flags a branch switch
    max_joint_jump_rad: float = 1.5
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_shapes(self) -> RobotConfig:
        if len(self.home_qpos) != self.dof:
            raise ValueError(f"home_qpos has {len(self.home_qpos)} entries, dof is {self.dof}")
        T = np.asarray(self.T_E_TCP, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_E_TCP must be 4x4, got {T.shape}")
        if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("T_E_TCP bottom row must be [0, 0, 0, 1]")
        if self.position_limits and len(self.position_limits) != self.dof:
            raise ValueError("position_limits length must equal dof")
        if self.home_gripper_qpos and len(self.home_gripper_qpos) != self.gripper_dof:
            raise ValueError("home_gripper_qpos length must equal gripper_dof")
        if self.gripper_joint_names and len(self.gripper_joint_names) != self.gripper_dof:
            raise ValueError("gripper_joint_names length must equal gripper_dof")
        return self

    @property
    def T_E_TCP_array(self) -> NDArray[np.float64]:
        return np.asarray(self.T_E_TCP, dtype=np.float64)


class QualityMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    hand_missing_ratio: float = 0.0
    ik_success_ratio: float = 0.0
    p95_position_error_m: float = 0.0
    p95_rotation_error_rad: float = 0.0
    robot_visible_ratio: float = 0.0
    fallback_frame_ratio: float = 0.0
    max_joint_jump_rad: float = 0.0
    mean_reach_ratio: float = 0.0
    hit_rate_ratio: float = 0.0
    so3_max_correction_rad: float = 0.0
    non_finite_ratio: float = 0.0


class QualityReport(BaseModel):
    """``ego2robot.quality.v1`` (system-design.md 6.6)."""

    model_config = SCHEMA_CONFIG

    schema_version: str = "ego2robot.quality.v1"
    episode_id: str
    valid: bool = False
    frame_valid_ratio: float = 0.0
    metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    reasons: list[str] = Field(default_factory=list)
    frame_reasons: dict[str, int] = Field(default_factory=dict)
    threshold_profile: str = "repro_v1"
    gate: str = "M3"
    stages_verified: list[str] = Field(default_factory=list)

    def metric_dict(self) -> dict[str, float]:
        return self.metrics.model_dump()


class StageManifest(BaseModel):
    """``stage-manifest.json`` (system-design.md 12)."""

    model_config = SCHEMA_CONFIG

    schema_version: str = "ego2robot.stage_manifest.v1"
    stage: str
    stage_version: str
    status: str = "pending"
    input_fingerprints: dict[str, str] = Field(default_factory=dict)
    config_hash: str = ""
    code_revision: str = ""
    started_at: str = ""
    duration_s: float = 0.0
    outputs: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    error: str | None = None
