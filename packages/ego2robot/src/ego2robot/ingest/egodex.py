"""EgoDex Path A adapter: paired MP4 + HDF5 annotations -> canonical ``HandSequence``.

Source of truth for the file layout: the official repository
https://github.com/apple/ml-egodex (``README.md`` and ``utils/skeleton_tfs.py``),
a copy of which is kept in ``docs/references/egodex/``.

Facts the adapter relies on
---------------------------
* Every HDF5 file holds ``/transforms/<joint>`` with shape ``[N, 4, 4]`` and
  ``/camera/intrinsic`` with shape ``[3, 3]``.
* ``/confidences/<joint>`` (shape ``[N]``) exists in *most* files; zero means the
  joint is occluded or undetected.
* All transforms, **including the camera extrinsics**, are expressed in the
  ARKit origin frame.  Apple's ``utils/data_utils.convert_to_camera_frame``
  converts with ``inv(cam_ext) @ tf``, which makes ``/transforms/camera``
  ``T_W_C`` in this project's naming: it maps camera-frame points into the ARKit
  world frame.
* The annotation is a *skeletal transform tree*, not a MANO parameter set.  Only
  the joints that the retargeting maths actually consumes are required: wrist,
  thumb tip, index tip and middle tip (system-design.md 7).

Because the ARKit skeleton has 24 hand joints (a metacarpal plus four phalanges
per finger) while this project's persisted contract is MANO-21, joints that
cannot be derived from real ARKit nodes are written as ``NaN`` with confidence
``0`` instead of being fabricated (ADR-0003).  The four joints the retargeter
needs are always real measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.frames import Frame, TransformName
from ..core.schema import CameraModel, HandSequence
from ..core.transforms import compose_transforms, invert_transform, transform_points
from .fixture import HandAnnotations

__all__ = [
    "ARKIT_HAND_JOINTS",
    "EGODEX_DEFAULT_CAMERA",
    "EgoDexHandAdapter",
    "EgoDexProbe",
    "MANO21_JOINT_NAMES",
    "MANO21_SOURCE_INDEX",
    "handedness_sign",
    "probe_egodex_hdf5",
]

# ---------------------------------------------------------------------------
# Joint vocabulary
# ---------------------------------------------------------------------------

#: ARKit hand joint order as published in ``utils/skeleton_tfs.py``.
ARKIT_HAND_JOINTS: dict[str, list[str]] = {
    "left": [
        "leftLittleFingerMetacarpal",
        "leftLittleFingerKnuckle",
        "leftLittleFingerIntermediateBase",
        "leftLittleFingerIntermediateTip",
        "leftLittleFingerTip",
        "leftRingFingerMetacarpal",
        "leftRingFingerKnuckle",
        "leftRingFingerIntermediateBase",
        "leftRingFingerIntermediateTip",
        "leftRingFingerTip",
        "leftMiddleFingerMetacarpal",
        "leftMiddleFingerKnuckle",
        "leftMiddleFingerIntermediateBase",
        "leftMiddleFingerIntermediateTip",
        "leftMiddleFingerTip",
        "leftIndexFingerMetacarpal",
        "leftIndexFingerKnuckle",
        "leftIndexFingerIntermediateBase",
        "leftIndexFingerIntermediateTip",
        "leftIndexFingerTip",
        "leftThumbKnuckle",
        "leftThumbIntermediateBase",
        "leftThumbIntermediateTip",
        "leftThumbTip",
    ],
    "right": [
        "rightLittleFingerMetacarpal",
        "rightLittleFingerKnuckle",
        "rightLittleFingerIntermediateBase",
        "rightLittleFingerIntermediateTip",
        "rightLittleFingerTip",
        "rightRingFingerMetacarpal",
        "rightRingFingerKnuckle",
        "rightRingFingerIntermediateBase",
        "rightRingFingerIntermediateTip",
        "rightRingFingerTip",
        "rightMiddleFingerMetacarpal",
        "rightMiddleFingerKnuckle",
        "rightMiddleFingerIntermediateBase",
        "rightMiddleFingerIntermediateTip",
        "rightMiddleFingerTip",
        "rightIndexFingerMetacarpal",
        "rightIndexFingerKnuckle",
        "rightIndexFingerIntermediateBase",
        "rightIndexFingerIntermediateTip",
        "rightIndexFingerTip",
        "rightThumbKnuckle",
        "rightThumbIntermediateBase",
        "rightThumbIntermediateTip",
        "rightThumbTip",
    ],
}

#: MANO-21 joint names in the canonical order persisted by this project.
MANO21_JOINT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
)

#: MANO-21 index -> dotted path into the EgoDex HDF5 transform tree, or ``None``.
#: ``None`` means "no ARKit node exists for this MANO joint" (see ADR-0003).
MANO21_SOURCE_INDEX: dict[int, str | None] = {
    0: "hand",  # wrist
    1: "thumbKnuckle",
    2: "thumbKnuckle",
    3: "thumbIntermediateTip",
    4: "thumbTip",
    5: None,  # index metacarpal: absent from MANO-21
    6: "indexFingerKnuckle",
    7: "indexFingerIntermediateTip",
    8: "indexFingerTip",
    9: None,
    10: "middleFingerKnuckle",
    11: "middleFingerIntermediateTip",
    12: "middleFingerTip",
    13: None,
    14: "ringFingerKnuckle",
    15: "ringFingerIntermediateTip",
    16: "ringFingerTip",
    17: None,
    18: "littleFingerKnuckle",
    19: "littleFingerIntermediateTip",
    20: "littleFingerTip",
}

#: EgoDex serves 1080p; Apple reports these intrinsics as constant for the set.
EGODEX_DEFAULT_CAMERA = {
    "width": 1920,
    "height": 1080,
    "K": [[736.6339, 0.0, 960.0], [0.0, 736.6339, 540.0], [0.0, 0.0, 1.0]],
}


def handedness_sign(handedness: int) -> int:
    """[PAPER] appendix A.3: ``s = +1`` right hand, ``s = -1`` left hand."""
    if handedness > 0:
        return 1
    if handedness < 0:
        return -1
    raise ValueError("handedness 0 (unknown) cannot define a grasp-axis sign")


# ---------------------------------------------------------------------------
# Per-hand reading helpers (ADR-0021)
# ---------------------------------------------------------------------------


def _handedness_for_side(side: str) -> int:
    if side == "right":
        return 1
    if side == "left":
        return -1
    raise ValueError(f"unknown hand side {side!r}; expected 'left' or 'right'")


def _side_transform_key(side: str, suffix: str) -> str:
    """``('right', 'hand') -> 'rightHand'``, ``('left', 'thumbTip') -> 'leftThumbTip'``."""
    if suffix == "hand":
        return f"{side}Hand"
    return f"{side}{suffix[0].upper()}{suffix[1:]}"


@dataclass
class _SideReading:
    """One hand's raw ARKit readings, before the world -> camera conversion."""

    side: str
    keypoints_W_m: np.ndarray  # [n, 21, 3] float32, NaN where no ARKit node exists
    keypoint_confidence: np.ndarray  # [n, 21] float32
    available_joint_names: list[str]
    derived_mano_indices: list[int]
    missing_required: list[str]
    confidence_source: str


#: The four ARKit suffixes the retargeter consumes, keyed by MANO index.
_RETARGET_SOURCE: dict[int, str] = {
    0: "hand",
    4: "thumbTip",
    8: "indexFingerTip",
    12: "middleFingerTip",
}


def _read_side(tfs: Any, cgrp: Any, side: str, sl: slice, n: int) -> _SideReading:
    """Read one hand's transform tree and confidence into MANO-21 arrays."""
    keypoints = np.full((n, 21, 3), np.nan, dtype=np.float32)
    keypoint_conf = np.zeros((n, 21), dtype=np.float32)
    available_joint_names: list[str] = []
    derived_indices: list[int] = []
    for mano_idx, suffix in MANO21_SOURCE_INDEX.items():
        if suffix is None:
            continue
        key = _side_transform_key(side, suffix)
        if key not in tfs:
            continue
        T_W_J = np.asarray(tfs[key][sl], dtype=np.float64)
        available_joint_names.append(key)
        if mano_idx not in _RETARGET_SOURCE:
            derived_indices.append(mano_idx)
        keypoints[:, mano_idx, :] = T_W_J[:, :3, 3].astype(np.float32)

    missing_required = [
        _side_transform_key(side, suffix)
        for suffix in _RETARGET_SOURCE.values()
        if _side_transform_key(side, suffix) not in tfs
    ]

    conf_source = "absent"
    if cgrp is not None:
        wrist_key = _side_transform_key(side, "hand")
        if wrist_key in cgrp:
            keypoint_conf[:, 0] = np.asarray(cgrp[wrist_key][sl], dtype=np.float32)
            conf_source = "hand_joint_confidence"
        for mano_idx, suffix in MANO21_SOURCE_INDEX.items():
            if suffix is None or mano_idx == 0:
                continue
            jkey = _side_transform_key(side, suffix)
            if jkey in cgrp:
                keypoint_conf[:, mano_idx] = np.asarray(cgrp[jkey][sl], dtype=np.float32)
    else:
        conf_source = "no_confidences_group"
        keypoint_conf[:] = 1.0

    return _SideReading(
        side=side,
        keypoints_W_m=keypoints,
        keypoint_confidence=keypoint_conf,
        available_joint_names=available_joint_names,
        derived_mano_indices=derived_indices,
        missing_required=missing_required,
        confidence_source=conf_source,
    )


# ---------------------------------------------------------------------------
# HDF5 probing
# ---------------------------------------------------------------------------


@dataclass
class EgoDexProbe:
    """What an EgoDex episode file actually contains."""

    hdf5_path: Path
    num_frames: int
    joint_names: list[str] = field(default_factory=list)
    confidence_joints: list[str] = field(default_factory=list)
    camera_intrinsic: np.ndarray | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)

    @property
    def instruction(self) -> str:
        a = self.attrs
        if a.get("llm_type") == "reversible":
            which = str(a.get("which_llm_description", "1"))
            key = "llm_description" if which == "1" else "llm_description2"
            return str(a.get(key, a.get("llm_description", "")))
        return str(a.get("llm_description", ""))

    def summary(self) -> dict[str, Any]:
        return {
            "hdf5": str(self.hdf5_path),
            "num_frames": self.num_frames,
            "num_joints": len(self.joint_names),
            "num_confidences": len(self.confidence_joints),
            "has_camera_intrinsic": self.camera_intrinsic is not None,
            "missing_required_joints": list(self.missing_required),
            "instruction": self.instruction,
        }


def probe_egodex_hdf5(path: str | Path) -> EgoDexProbe:
    """Inspect an EgoDex HDF5 file without loading the heavy arrays.

    Raises ``ValueError`` when the file has no ``/transforms`` group, i.e. it is
    not an EgoDex episode.  Use :func:`is_egodex_hdf5` to test first.
    """
    import h5py

    p = Path(path)
    with h5py.File(p, "r") as fh:
        if "transforms" not in fh:
            raise ValueError(
                f"{p}: no '/transforms' group; is this an EgoDex file? "
                f"top-level keys: {sorted(fh.keys())}"
            )
        joints = sorted(fh["transforms"].keys())
        confs = sorted(fh["confidences"].keys()) if "confidences" in fh else []
        intr = np.asarray(fh["camera/intrinsic"][()], dtype=np.float64) if "camera/intrinsic" in fh else None
        attrs = {k: fh.attrs[k] for k in fh.attrs}
        n = int(fh["transforms"][joints[0]].shape[0]) if joints else 0

    required = ["camera", "leftHand", "rightHand"]
    missing = [j for j in required if j not in joints]
    return EgoDexProbe(
        hdf5_path=p,
        num_frames=n,
        joint_names=joints,
        confidence_joints=confs,
        camera_intrinsic=intr,
        attrs=attrs,
        missing_required=missing,
    )


def is_egodex_hdf5(path: str | Path) -> bool:
    """True when the file carries an EgoDex-style ``/transforms`` tree."""
    import h5py

    with h5py.File(path, "r") as fh:
        if "transforms" not in fh:
            return False
        return "camera" in fh["transforms"]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class EgoDexHandAdapter:
    """Path A provider: EgoDex HDF5 annotations -> :class:`HandSequence`.

    ``frame_convention`` selects how ARKit world-frame transforms are mapped into
    this project's camera optical frame:

    ``"inverse_camera_pose"`` (default)
        ``T_C_* = inverse(T_W_C) @ T_W_*`` -- the convention used by Apple's own
        ``utils/data_utils.convert_to_camera_frame``.
    ``"transpose_camera_pose"``
        the same product with the rotation transposed instead of inverted, kept
        because a mislabelled extrinsics convention is otherwise indistinguishable
        from correct data (see ``docs/decisions/0003``).
    """

    hand: str = "right"
    #: ordered hand sides for a multi-hand read; ``None`` keeps the single-hand
    #: behaviour selected by ``hand`` (ADR-0021).  ``(right, left)`` is the Path A
    #: order and becomes ``hand_ids`` on the persisted sequence.
    hands: tuple[str, ...] | None = None
    frame_convention: str = "inverse_camera_pose"
    min_confidence: float = 0.0
    #: [PAPER] Path B constrains hand depth to [0.05, 0.4] m; Path A reports the
    #: measured range rather than clipping, so violations surface in QC instead.
    clip_depth_range_m: tuple[float, float] | None = None

    name: str = "egodex_hdf5"
    version: str = "2"

    def __post_init__(self) -> None:
        if self.hand not in ARKIT_HAND_JOINTS:
            raise ValueError(f"hand must be 'left' or 'right', got {self.hand!r}")
        if self.hands is not None:
            normalized = tuple(str(s) for s in self.hands)
            if not normalized:
                raise ValueError("hands must not be empty when provided")
            unknown = [s for s in normalized if s not in ARKIT_HAND_JOINTS]
            if unknown:
                raise ValueError(f"hands contains unknown side(s) {unknown}")
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"hands must not repeat a side, got {normalized}")
            self.hands = normalized
        elif self.hand not in ARKIT_HAND_JOINTS:  # pragma: no cover - kept explicit
            raise ValueError(f"hand must be 'left' or 'right', got {self.hand!r}")
        if self.frame_convention not in {"inverse_camera_pose", "transpose_camera_pose"}:
            raise ValueError(
                "frame_convention must be 'inverse_camera_pose' or 'transpose_camera_pose'"
            )

    # -- helpers -----------------------------------------------------------

    @property
    def sides(self) -> list[str]:
        """Hand sides in persisted ``H`` order (one entry for the single-hand path)."""
        return list(self.hands) if self.hands else [self.hand]

    @property
    def hand_ids(self) -> list[str]:
        return self.sides

    @classmethod
    def bimanual(
        cls, sides: tuple[str, ...] = ("right", "left"), **kwargs: Any
    ) -> "EgoDexHandAdapter":
        """Adapter that reads both hands into one ``[T, H, ...]`` sequence."""
        return cls(hands=tuple(sides), **kwargs)

    @property
    def handedness(self) -> int:
        return handedness_sign(_handedness_for_side(self.hand))

    def _transform_key(self, suffix: str) -> str:
        return _side_transform_key(self.hand, suffix)

    def camera_model(self, probe: EgoDexProbe, *, width: int | None = None, height: int | None = None) -> CameraModel:
        """Build a :class:`CameraModel` labelled ``dataset`` (never ``assumed``)."""
        K = probe.camera_intrinsic
        if K is None:
            K = np.asarray(EGODEX_DEFAULT_CAMERA["K"], dtype=np.float64)
        return CameraModel(
            width=width or EGODEX_DEFAULT_CAMERA["width"],
            height=height or EGODEX_DEFAULT_CAMERA["height"],
            K=np.asarray(K, dtype=np.float64).tolist(),
            distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
            calibration_source="dataset",
            confidence=0.9,
        )

    def _camera_to_world(self, T_W_C_raw: np.ndarray) -> np.ndarray:
        """Interpret ``/transforms/camera`` as ``T_W_C``."""
        if self.frame_convention == "inverse_camera_pose":
            return np.asarray(T_W_C_raw, dtype=np.float64)
        T = np.asarray(T_W_C_raw, dtype=np.float64).copy()
        T[:3, :3] = T[:3, :3].T
        return T

    # -- main entry point --------------------------------------------------

    def load(
        self,
        hdf5_path: str | Path,
        *,
        num_frames: int | None = None,
        frame_offset: int = 0,
        camera: CameraModel | None = None,
    ) -> tuple[HandSequence, dict[str, Any]]:
        """Read one or both hands and return ``(HandSequence, provenance)``.

        The returned provenance dict records every conversion that was applied,
        so a reviewer can reconstruct exactly how the numbers were produced
        (implementation-plan.md Gate M1 forbids silent zero-filling).

        ADR-0021: with ``hands=(right, left)`` the arrays gain the existing
        ``[T, H, ...]`` hand axis in exactly that order.  Both hands share the
        decoded range, the world-to-camera transform and the PTS grid; a missing
        retarget joint on *either* side fails the read instead of writing a
        half-empty hand.
        """
        import h5py

        p = Path(hdf5_path)
        if not p.is_file():
            raise FileNotFoundError(f"EgoDex HDF5 not found: {p}")

        sides = self.sides
        pose_name = TransformName(source=Frame.C, target=Frame.W)
        with h5py.File(p, "r") as fh:
            if "transforms" not in fh:
                raise ValueError(f"{p}: no '/transforms' group; is this an EgoDex file?")
            tfs = fh["transforms"]
            if "camera" not in tfs:
                raise ValueError(f"{p}: no '/transforms/camera'; cannot build the camera frame")

            n_total = int(tfs["camera"].shape[0])
            n = n_total if num_frames is None else min(num_frames, n_total - frame_offset)
            if n <= 0:
                raise ValueError(f"{p}: requested {num_frames} frames from offset {frame_offset} of {n_total}")
            sl = slice(frame_offset, frame_offset + n)

            T_W_C_raw = np.asarray(tfs["camera"][sl], dtype=np.float64)
            camera_available = True

            # --- every requested side's joints and confidences --------------
            cgrp = fh["confidences"] if "confidences" in fh else None
            readings = [_read_side(tfs, cgrp, side, sl, n) for side in sides]
            missing = {
                r.side: r.missing_required for r in readings if r.missing_required
            }
            if missing:
                raise ValueError(
                    f"{p}: missing transform(s) required by the retargeter for hand(s) "
                    f"{sorted(missing)}: {missing}. Available: {sorted(tfs.keys())[:12]}..."
                )
            attrs = {k: fh.attrs[k] for k in fh.attrs}

        # --- world -> camera optical frame (shared by every hand) ---------
        if self.frame_convention == "inverse_camera_pose":
            T_C_W = np.stack([invert_transform(T_W_C_raw[i]) for i in range(n)], axis=0)
        else:
            T_C_W = np.empty_like(T_W_C_raw)
            for i in range(n):
                Ti = T_W_C_raw[i].copy()
                Ti[:3, :3] = Ti[:3, :3].T
                T_C_W[i] = invert_transform(Ti)

        cam = camera or self.camera_model(EgoDexProbe(hdf5_path=p, num_frames=n, camera_intrinsic=None))

        keypoints_per_side: list[np.ndarray] = []
        uv_per_side: list[np.ndarray] = []
        conf_per_side: list[np.ndarray] = []
        detection_per_side: list[np.ndarray] = []
        valid_per_side: list[np.ndarray] = []
        projection_checks: dict[str, dict[str, Any]] = {}
        depth_all: list[np.ndarray] = []
        inside_counts: list[tuple[int, int]] = []

        for reading in readings:
            keypoints_C = np.full_like(reading.keypoints_W_m, np.nan)
            for i in range(n):
                pts = reading.keypoints_W_m[i]
                finite = np.isfinite(pts).all(axis=1)
                if finite.any():
                    keypoints_C[i, finite] = transform_points(T_C_W[i], pts[finite]).astype(np.float32)
            uv = cam.project(keypoints_C)

            # --- projection self-check -----------------------------------
            depth = keypoints_C[:, :, 2]
            finite_depth = depth[np.isfinite(depth)]
            depth_all.append(finite_depth)
            depth_min = float(np.min(finite_depth)) if finite_depth.size else float("nan")
            depth_max = float(np.max(finite_depth)) if finite_depth.size else float("nan")

            inside = (
                (uv[:, :, 0] >= 0)
                & (uv[:, :, 0] < cam.width)
                & (uv[:, :, 1] >= 0)
                & (uv[:, :, 1] < cam.height)
            )
            inside_valid = inside & np.isfinite(uv[:, :, 0])
            n_valid_pts = int(np.count_nonzero(np.isfinite(uv[:, :, 0])))
            n_inside = int(np.count_nonzero(inside_valid))
            inside_ratio = float(n_inside) / max(1, n_valid_pts)
            inside_counts.append((n_inside, n_valid_pts))

            # --- validity (per hand, same policy as the single-hand path) --
            wrist_conf = reading.keypoint_confidence[:, 0]
            detection_confidence = wrist_conf.copy()
            if reading.confidence_source == "no_confidences_group":
                detection_confidence = np.ones(n, dtype=np.float32)
            valid = (detection_confidence >= self.min_confidence) & np.isfinite(
                keypoints_C[:, 0, :]
            ).all(axis=1)
            if self.clip_depth_range_m is not None:
                lo, hi = self.clip_depth_range_m
                valid &= (depth[:, 0] >= lo) & (depth[:, 0] <= hi)

            keypoints_per_side.append(keypoints_C)
            uv_per_side.append(uv)
            conf_per_side.append(reading.keypoint_confidence)
            detection_per_side.append(detection_confidence)
            valid_per_side.append(valid)
            projection_checks[reading.side] = {
                "inside_image_ratio": round(inside_ratio, 6),
                "depth_min_m": round(depth_min, 6) if np.isfinite(depth_min) else None,
                "depth_max_m": round(depth_max, 6) if np.isfinite(depth_max) else None,
            }

        keypoints_C_all = np.stack(keypoints_per_side, axis=1).astype(np.float32)
        uv_all = np.stack(uv_per_side, axis=1).astype(np.float32)
        conf_all = np.stack(conf_per_side, axis=1).astype(np.float32)
        detection_all = np.stack(detection_per_side, axis=1).astype(np.float32)
        valid_all = np.stack(valid_per_side, axis=1)

        unique_sources = {r.confidence_source for r in readings}
        conf_source = unique_sources.pop() if len(unique_sources) == 1 else "mixed"

        timestamp_s = np.arange(n, dtype=np.float64) / 30.0
        frame_index = np.arange(n, dtype=np.int64) + frame_offset

        n_hands = len(sides)
        handedness = np.stack(
            [np.full(n, _handedness_for_side(side), dtype=np.int8) for side in sides], axis=1
        )
        seq = HandSequence(
            {
                "timestamp_s": timestamp_s,
                "frame_index": frame_index,
                "keypoints_C_m": keypoints_C_all,
                "keypoints_uv_px": uv_all,
                "handedness": handedness,
                "detection_confidence": detection_all,
                "valid": valid_all,
                "keypoint_confidence": conf_all,
            },
            meta={
                "provider": self.name,
                "provider_version": self.version,
                "source_hdf5": str(p),
                "hand": sides[0] if n_hands == 1 else None,
                "hands": list(sides),
                "hand_ids": list(sides),
                "frame_convention": self.frame_convention,
                "pose_variable": str(pose_name),
                "source_joint_order": "arkit24",
                "target_joint_order": "mano21",
                "available_source_joints": sorted(
                    {name for r in readings for name in r.available_joint_names}
                ),
                "derived_mano_indices": sorted(
                    {int(i) for r in readings for i in r.derived_mano_indices}
                ),
                "nan_mano_indices": [i for i, s in MANO21_SOURCE_INDEX.items() if s is None],
                "confidence_source": conf_source,
            },
        )

        finite_all = np.concatenate(depth_all) if depth_all else np.zeros(0)
        finite_all = finite_all[np.isfinite(finite_all)]
        total_inside = sum(c[0] for c in inside_counts)
        total_points = sum(c[1] for c in inside_counts)
        provenance: dict[str, Any] = {
            "provider": self.name,
            "provider_version": self.version,
            "source_hdf5": str(p),
            "num_frames_total": n_total,
            "num_frames_used": n,
            "frame_offset": frame_offset,
            "num_hands": n_hands,
            "hands": list(sides),
            "hand_ids": list(sides),
            "hand": sides[0] if n_hands == 1 else None,
            "handedness_sign": handedness_sign(_handedness_for_side(sides[0])),
            "handedness_signs": {side: handedness_sign(_handedness_for_side(side)) for side in sides},
            "frame_convention": self.frame_convention,
            "world_to_camera_transform": str(TransformName(source=Frame.W, target=Frame.C)),
            "source_frame": "W (ARKit origin)",
            "target_frame": "C (camera optical)",
            "length_unit_in": "m",
            "length_unit_out": "m",
            "unit_conversion": "none (ARKit skeleton is already metric)",
            "source_joint_order": "arkit24",
            "target_joint_order": "mano21",
            "nan_mano_indices": [i for i, s in MANO21_SOURCE_INDEX.items() if s is None],
            "confidence_source": conf_source,
            "confidence_by_hand": {r.side: r.confidence_source for r in readings},
            "instruction": _extract_instruction(attrs),
            "hdf5_attributes": {k: str(v) for k, v in attrs.items()},
            "projection_check": {
                "inside_image_ratio": round(total_inside / max(1, total_points), 6),
                "depth_min_m": round(float(np.min(finite_all)), 6) if finite_all.size else None,
                "depth_max_m": round(float(np.max(finite_all)), 6) if finite_all.size else None,
                "per_hand": projection_checks,
                "note": (
                    "EgoDex documents a perspective mismatch between the synthesized "
                    "RGB and the annotation, so a residual reprojection offset is "
                    "expected (ml-egodex README, visualize_2d.py)."
                ),
                "statistics_scope": (
                    "single hand" if n_hands == 1 else f"both hands pooled over {n_hands} hands"
                ),
            },
            "camera_available_in_hdf5": camera_available,
            "per_hand": {
                r.side: {
                    "handedness_sign": handedness_sign(_handedness_for_side(r.side)),
                    "available_source_joints": sorted(r.available_joint_names),
                    "derived_mano_indices": sorted(r.derived_mano_indices),
                    "confidence_source": r.confidence_source,
                    "valid_frames": int(np.count_nonzero(valid_all[:, i])),
                    "valid_ratio": round(float(np.mean(valid_all[:, i])), 6),
                    "projection_check": projection_checks[r.side],
                }
                for i, r in enumerate(readings)
            },
        }
        return seq, provenance


def _extract_instruction(attrs: Mapping[str, Any]) -> str:
    if attrs.get("llm_type") == "reversible":
        which = str(attrs.get("which_llm_description", "1"))
        key = "llm_description" if which == "1" else "llm_description2"
        return str(attrs.get(key, attrs.get("llm_description", "")))
    return str(attrs.get("llm_description", ""))


def annotations_from_sequence(seq: HandSequence, hand_index: int = 0) -> HandAnnotations:
    """Down-convert a :class:`HandSequence` to the fixture annotation container."""
    return HandAnnotations(
        keypoints_C_m=np.asarray(seq["keypoints_C_m"][:, hand_index], dtype=np.float32),
        confidence=np.asarray(seq["detection_confidence"][:, hand_index], dtype=np.float32),
        handedness=np.asarray(seq["handedness"][:, hand_index], dtype=np.int8),
        uv_px=np.asarray(seq["keypoints_uv_px"][:, hand_index], dtype=np.float32),
        keypoint_confidence=np.asarray(seq["keypoint_confidence"], dtype=np.float32)
        if "keypoint_confidence" in seq
        else None,
        attributes=dict(seq.meta),
    )


def _compose_chain(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Small helper kept for callers that walk the ARKit hierarchy explicitly."""
    return compose_transforms(*transforms) if transforms else np.eye(4)
