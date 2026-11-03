"""Coordinate frame registry and naming rules.

System design 5.1 / 5.2.  Every persisted pose MUST name both frames via
``T_A_B`` ("maps points expressed in B into A").  This module is the single
place where frame semantics live so no module has to guess.

Frames
------
======  ===========================================================
 name    meaning
======  ===========================================================
 ``C``   camera optical frame, +x right, +y down, +z forward
 ``W``   optional scene world frame; never fabricated without calibration
 ``B``   robot base frame
 ``E``   robot end-effector link frame (MJCF body frame)
 ``TCP`` canonical parallel-gripper tool centre point / grasp frame
 ``H``   input hand / MANO local frame; adapter-internal only
======  ===========================================================

Persisted boundary conventions (system-design.md 5.2):

* length: metres
* time: seconds, monotonically increasing
* angle: radians
* quaternion: ``[qx, qy, qz, qw]`` (xyzw)
* image: RGB (OpenCV BGR only inside adapters)
* depth: metres; non-metric depth must be labelled ``relative``
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Frame",
    "FrameConvention",
    "TransformName",
    "parse_transform_name",
    "transform_name",
    "CAMERA_OPTICAL_CONVENTION",
    "ROBOT_BASE_CONVENTION",
    "TCP_CONVENTION",
]


class Frame(str, Enum):
    """Named coordinate frames used across the pipeline."""

    C = "C"
    W = "W"
    B = "B"
    E = "E"
    TCP = "TCP"
    H = "H"


@dataclass(frozen=True)
class FrameConvention:
    """Human-readable semantics for a frame."""

    frame: Frame
    description: str
    x_axis: str
    y_axis: str
    z_axis: str
    handedness: str = "right"

    def describe(self) -> str:
        return (
            f"{self.frame.value}: {self.description} "
            f"[x={self.x_axis}, y={self.y_axis}, z={self.z_axis}, {self.handedness}-handed]"
        )


CAMERA_OPTICAL_CONVENTION = FrameConvention(
    frame=Frame.C,
    description="camera optical frame (projection frame of the pinhole model)",
    x_axis="image right",
    y_axis="image down",
    z_axis="forward, along the optical axis",
)

WORLD_CONVENTION = FrameConvention(
    frame=Frame.W,
    description="optional scene world frame; never fabricated without calibration",
    x_axis="calibration defined",
    y_axis="calibration defined",
    z_axis="calibration defined",
)

ROBOT_BASE_CONVENTION = FrameConvention(
    frame=Frame.B,
    description="robot base frame (MJCF world body of the robot model)",
    x_axis="model defined",
    y_axis="model defined",
    z_axis="model defined (Franka: +z up)",
)

EEF_CONVENTION = FrameConvention(
    frame=Frame.E,
    description="robot end-effector link frame (MJCF body frame, Franka: 'hand')",
    x_axis="model defined",
    y_axis="model defined",
    z_axis="model defined",
)

TCP_CONVENTION = FrameConvention(
    frame=Frame.TCP,
    description="canonical parallel-gripper grasp frame",
    x_axis="approach direction",
    y_axis="gripper normal (jaw-plane normal)",
    z_axis="grasp axis, along the jaw line (thumb tip -> virtual fingertip)",
)

HAND_CONVENTION = FrameConvention(
    frame=Frame.H,
    description="input hand / MANO local frame; adapter-internal only",
    x_axis="model defined",
    y_axis="model defined",
    z_axis="model defined",
)

CONVENTIONS: dict[Frame, FrameConvention] = {
    Frame.C: CAMERA_OPTICAL_CONVENTION,
    Frame.W: WORLD_CONVENTION,
    Frame.B: ROBOT_BASE_CONVENTION,
    Frame.E: EEF_CONVENTION,
    Frame.TCP: TCP_CONVENTION,
    Frame.H: HAND_CONVENTION,
}


@dataclass(frozen=True)
class TransformName:
    """A parsed ``T_A_B`` transform name.

    ``T_A_B`` maps a point expressed in ``B`` into ``A``::

        p_A = T_A_B @ p_B
    """

    source: Frame
    target: Frame

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"T_{self.source.value}_{self.target.value}"

    @property
    def inverse(self) -> TransformName:
        return TransformName(source=self.target, target=self.source)

    def describe(self) -> str:
        return f"{self} maps points from {self.target.value} into {self.source.value}"


def parse_transform_name(name: str) -> TransformName:
    """Parse ``T_A_B`` into a :class:`TransformName`.

    Raises ``ValueError`` for anything that is not an explicit two-frame name,
    so that the banned bare ``pose``/``transform`` variables cannot sneak in.
    """
    text = name.strip()
    if not text.startswith("T_"):
        raise ValueError(f"transform name must start with 'T_': got {name!r}")
    parts = text.split("_")
    if len(parts) != 3:
        raise ValueError(f"transform name must look like T_A_B with explicit frames: got {name!r}")
    source, target = parts[1], parts[2]
    try:
        return TransformName(source=Frame(source), target=Frame(target))
    except ValueError as exc:  # unknown frame letter
        known = ", ".join(f.value for f in Frame)
        raise ValueError(f"unknown frame in {name!r}; known frames: {known}") from exc


def transform_name(source: Frame | str, target: Frame | str) -> TransformName:
    """Build a :class:`TransformName` from two frames."""
    return TransformName(source=Frame(source), target=Frame(target))
