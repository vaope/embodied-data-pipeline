"""``HandPoseProvider`` protocol: the seam between path A and path B.

System design 4.1 requires that both input paths converge on the same
:class:`~ego2robot.core.schema.HandSequence` and that downstream modules cannot
tell which model produced it.  Everything robot-specific is forbidden here:
perception must never import ``robots`` or ``mujoco``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ...core.schema import CameraModel, HandSequence

__all__ = ["HandPoseProvider", "HandSourceInfo"]


@dataclass
class HandSourceInfo:
    """Where hand poses came from, recorded in the stage manifest."""

    provider: str
    version: str = ""
    path: str = "A"
    source_files: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HandPoseProvider(Protocol):
    """Produces a canonical :class:`HandSequence` from an ego clip."""

    name: str
    version: str

    def load(
        self,
        source: str | Path,
        *,
        num_frames: int | None = None,
        frame_offset: int = 0,
        camera: CameraModel | None = None,
    ) -> tuple[HandSequence, dict[str, Any]]:
        """Return ``(sequence, provenance)``.

        ``provenance`` must record at minimum: the source frame, the target
        frame, the length unit conversion, the joint order mapping and the
        confidence source.  Returning zeros for missing joints is forbidden
        (implementation-plan.md Gate M1).
        """
        ...
