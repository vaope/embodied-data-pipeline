"""``RobotModel`` protocol: the seam between canonical actions and any morphology.

System design 8.1.  The canonical action sequence is robot independent; a robot
adapter is the only thing allowed to know about MJCF, joint names, IK solvers or
renderers.  ``perception`` and ``retarget`` must never import this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CollisionResult",
    "IKResult",
    "RenderFrame",
    "RobotModel",
    "SE3",
]

SE3 = NDArray[np.float64]


@dataclass
class IKResult:
    """Outcome of one IK solve.

    ``T_B_TCP_fk`` is always filled in, even on failure, so a reviewer can see
    *where* the solver ended up rather than only that it failed.
    """

    success: bool
    qpos: NDArray[np.float64]
    T_B_TCP_fk: SE3
    position_error_m: float
    rotation_error_rad: float
    iterations: int = 0
    status: str = ""
    status_code: int = 0
    joint_limit_hit: bool = False
    error: str | None = None


@dataclass
class CollisionResult:
    """Contact classification for one configuration."""

    has_self_collision: bool
    self_contact_pairs: list[tuple[str, str]] = field(default_factory=list)
    adjacent_contact_pairs: list[tuple[str, str]] = field(default_factory=list)
    scene_contact_pairs: list[tuple[str, str]] = field(default_factory=list)
    max_penetration_m: float = 0.0
    code: int = 0

    _NAMES = {0: "none", 1: "self_collision", 2: "scene_contact"}

    def describe_code(self) -> str:
        return self._NAMES.get(self.code, f"code_{self.code}")


@dataclass
class RenderFrame:
    """One rendered frame: RGB, masks and metric depth (system-design.md 6.5)."""

    rgb: NDArray[np.uint8]  # [H, W, 3]
    robot_mask: NDArray[np.bool_]  # [H, W] arm body only (gripper excluded)
    gripper_mask: NDArray[np.bool_]  # [H, W]
    depth_m: NDArray[np.float32]  # [H, W]; NaN where nothing was hit
    frame_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RobotModel(Protocol):
    """One robot morphology."""

    robot_id: str
    dof: int
    max_reach_m: float
    home_qpos: NDArray[np.float64]

    def forward_kinematics(self, qpos: NDArray[np.floating[Any]]) -> SE3:
        """``T_B_TCP`` for a joint configuration."""
        ...

    def solve_ik(
        self,
        target_T_B_TCP: SE3,
        q_init: NDArray[np.floating[Any]] | None = None,
    ) -> IKResult:
        """Solve for a ``T_B_TCP`` target, warm-starting from ``q_init``."""
        ...

    def check_collision(self, qpos: NDArray[np.floating[Any]]) -> CollisionResult:
        """Classify contacts for a configuration."""
        ...

    def render(self, qpos: NDArray[np.floating[Any]], camera: Any) -> RenderFrame:
        """Render RGB, masks and metric depth from a camera specification."""
        ...


#: signature of a pluggable IK backend (Mink, or a deterministic fallback)
IKSolverFn = Callable[[SE3, NDArray[np.float64] | None], IKResult]
