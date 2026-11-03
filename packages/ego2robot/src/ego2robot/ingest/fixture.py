"""Project fixture format: a self-describing ego clip plus metric 3D hand poses.

The fixtures this module reads are *project-owned*, not an Apple format.  They
exist because the official EgoDex test set is a ~16 GB download
(implementation-plan.md section 4) and because M0-M2 must be verifiable with
fully known geometry.  Each fixture directory looks like::

    <episode_dir>/
      episode.json          # ego2robot manifest (source, timebase, camera, task)
      source/
        video.mp4           # optional real footage; synthetic fixtures render one
        camera.json         # camera-K + optional T_C_W for synthetic rendering
        hands.hdf5          # /hands/<name>  float32 [T, 21, 3] metres, camera optical frame
                            # /hands/<name>_confidence float32 [T]
                            # /hands/<name>_handedness int8 [T]
                            # /hands/<name>_uv_px      float32 [T, 21, 2]  (self-check)
                            # attrs: frame="C", length_unit="m", keypoint_convention="mano21"

``EgoDexHandAdapter`` (``ingest/egodex.py``) is the other Path A provider; both
produce the same :class:`~ego2robot.core.schema.HandSequence`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "FIXTURE_FORMAT_VERSION",
    "HandAnnotations",
    "list_hands",
    "read_hand_annotations",
    "write_hand_annotations",
]

FIXTURE_FORMAT_VERSION = "ego2robot.fixture.hands.v1"


@dataclass
class HandAnnotations:
    """Per-hand annotation arrays read from a fixture HDF5 file."""

    keypoints_C_m: np.ndarray  # [T, 21, 3] float32, metres
    confidence: np.ndarray  # [T] float32
    handedness: np.ndarray  # [T] int8
    uv_px: np.ndarray | None  # [T, 21, 2] float32, optional self-check
    keypoint_confidence: np.ndarray | None  # [T, 21] float32, optional
    attributes: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.keypoints_C_m.shape[0])


def write_hand_annotations(  # pragma: no cover - exercised by the fixture builder
    path: str | Path,
    hands: Mapping[str, HandAnnotations],
    *,
    attributes: Mapping[str, Any] | None = None,
) -> Path:
    """Write fixture hand annotations to HDF5 (fixture builder side)."""
    import h5py

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as fh:
        grp = fh.create_group("hands")
        for name, ann in hands.items():
            h = grp.create_group(name)
            h.create_dataset("keypoints", data=np.asarray(ann.keypoints_C_m, dtype=np.float32))
            h.create_dataset("confidence", data=np.asarray(ann.confidence, dtype=np.float32))
            h.create_dataset("handedness", data=np.asarray(ann.handedness, dtype=np.int8))
            if ann.uv_px is not None:
                h.create_dataset("uv_px", data=np.asarray(ann.uv_px, dtype=np.float32))
            if ann.keypoint_confidence is not None:
                h.create_dataset(
                    "keypoint_confidence", data=np.asarray(ann.keypoint_confidence, dtype=np.float32)
                )
            for k, v in (attributes or {}).items():
                h.attrs[k] = v
            for k, v in ann.attributes.items():
                h.attrs[k] = v
        fh.attrs["fixture_format_version"] = FIXTURE_FORMAT_VERSION
        fh.attrs["frame"] = "C"
        fh.attrs["length_unit"] = "m"
        fh.attrs["keypoint_convention"] = "mano21"
    return p


def list_hands(path: str | Path) -> list[str]:
    """Names of the hand groups stored in a fixture HDF5 file."""
    import h5py

    with h5py.File(path, "r") as fh:
        if "hands" not in fh:
            raise ValueError(
                f"{path}: missing '/hands' group; found {sorted(fh.keys())}"
            )
        return sorted(fh["hands"].keys())


def read_hand_annotations(path: str | Path, hand: str) -> HandAnnotations:
    """Read one hand group, declaring frame and unit explicitly."""
    import h5py

    p = Path(path)
    with h5py.File(p, "r") as fh:
        if "hands" not in fh or hand not in fh["hands"]:
            raise ValueError(f"{p}: no hand group {hand!r}; available: {sorted(fh.get('hands', {}).keys())}")
        grp = fh["hands"][hand]
        attrs = dict(grp.attrs)
        frame = attrs.get("frame", "C")
        unit = attrs.get("length_unit", "m")
        keypoints = np.asarray(grp["keypoints"][()], dtype=np.float32)
        confidence = np.asarray(grp["confidence"][()], dtype=np.float32)
        handedness = np.asarray(grp["handedness"][()], dtype=np.int8)
        uv = np.asarray(grp["uv_px"][()], dtype=np.float32) if "uv_px" in grp else None
        kp_conf = (
            np.asarray(grp["keypoint_confidence"][()], dtype=np.float32)
            if "keypoint_confidence" in grp
            else None
        )

    if keypoints.ndim != 3 or keypoints.shape[1:] != (21, 3):
        raise ValueError(
            f"{p}:{hand}: keypoints must be [T, 21, 3], got {keypoints.shape}"
        )
    if frame != "C":
        raise ValueError(
            f"{p}:{hand}: annotations declare frame {frame!r}; the pipeline contract "
            "requires metric positions in the camera optical frame 'C' "
            "(system-design.md 6.2). Convert inside the adapter instead."
        )
    scale = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}.get(str(unit))
    if scale is None:
        raise ValueError(f"{p}:{hand}: unsupported length_unit {unit!r}")
    if scale != 1.0:
        keypoints = (keypoints.astype(np.float64) * scale).astype(np.float32)
        attrs["unit_conversion_applied"] = f"{unit} -> m (x{scale})"

    if confidence.shape != (keypoints.shape[0],):
        raise ValueError(f"{p}:{hand}: confidence shape {confidence.shape} != [{keypoints.shape[0]}]")
    if handedness.shape != (keypoints.shape[0],):
        raise ValueError(f"{p}:{hand}: handedness shape {handedness.shape} != [{keypoints.shape[0]}]")

    return HandAnnotations(
        keypoints_C_m=keypoints,
        confidence=confidence,
        handedness=handedness,
        uv_px=uv,
        keypoint_confidence=kp_conf,
        attributes=attrs,
    )
