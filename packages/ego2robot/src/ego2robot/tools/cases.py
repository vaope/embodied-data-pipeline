"""Validation case definitions: five synthetic ego clips with known geometry.

The fixture set is deliberately *small and analytically known*.  A real EgoDex
episode cannot tell you whether a wrong answer came from the retargeting maths,
the frame conversion or the annotation quality; these cases can, because every
case has a closed-form expected output (see
:mod:`ego2robot.tools.hand_synth`).

Case list
---------
``case1_right_translate``
    right hand, pick-move-place, pure +x/+z translation -- checks scale, axis
    order and the 7D delta signs.
``case2_left_mirror``
    the same motion with a left hand -- checks the handedness sign: both hands
    must produce the *same* canonical jaw axis direction.
``case3_static``
    a stationary hand at mid-depth -- relative action must be ~0 and the
    orientation must not drift.
``case4_rotation``
    a stationary wrist rotating 90 deg about the vertical axis -- checks the
    rotation-vector direction and magnitude.
``case5_degenerate``
    noise spikes, closed-jaw frames and a 14-frame dropout -- checks the
    two-round velocity filter, the degeneracy freeze and large-gap blending.

Every case writes::

    <fixture_root>/<episode_id>/
      episode.json          # e2r manifest with an explicit T_C_B
      source/video.mp4
      source/hands.hdf5
      source/camera.json
      case.json             # case definition incl. analytic ground truth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..core.transforms import (
    exp_so3,
    invert_transform,
    log_so3,
    make_transform,
    transform_points,
)
from .hand_synth import HandPose, expected_gripper_width, expected_tcp_frame, expected_tcp_position, make_hand_pose

__all__ = [
    "CASES_FILE",
    "CaseDefinition",
    "case_ground_truth",
    "load_cases",
]

CASES_FILE = "configs/fixtures/cases.yaml"


@dataclass
class CaseDefinition:
    """One validation case."""

    case_id: str
    instruction: str
    checks: str
    num_frames: int
    fps: float
    handedness: int
    width: int
    height: int
    sequence: Callable[[int], HandPose]
    T_C_B: np.ndarray
    corruptions: dict[str, Any] = field(default_factory=dict)
    object_xy_m: tuple[float, float] = (0.08, 0.10)
    notes: list[str] = field(default_factory=list)

    @property
    def episode_id(self) -> str:
        return self.case_id


def base_for_workspace(
    centroid_C_m: Sequence[float],
    *,
    reach_fraction: float = 0.26,
    robot_reach_m: float = 0.855,
    height_offset_m: float = 0.03,
    pitch_deg: float = 70.0,
) -> np.ndarray:
    """Place the robot base behind and below the workspace, facing it.

    ``T_C_B`` describes where the robot base *is*, expressed in the camera frame
    (``T_A_B`` is the pose of ``B`` in ``A``); the ego camera therefore sits at
    the origin of ``C``.

    Position: the base stands ``reach_fraction * robot_reach`` back from the
    workspace centroid along the horizontal direction away from the camera, and
    ``height_offset_m`` below it.

    Orientation::

        R = Rz(yaw) @ Rx(90 deg - pitch),  yaw = atan2(f_y, f_x) - 90 deg

    where ``f`` is the horizontal unit vector from the base to the workspace.
    That gives ``+x`` = base right, ``+y`` = base forward (pointing at the
    workspace) and ``+z`` = up, pitched forward by ``90 - pitch`` degrees.

    Note the sign inside ``Rx``: the rotation is applied *after* the yaw and in
    the *camera* frame, where ``+y`` points down, so a forward pitch is a
    rotation of ``+(90 - pitch)`` about camera ``+x``.  The opposite sign turns
    the base upside down, after which the arm lives in front of the camera and
    renders as an empty frame.  The invariant below catches that.

    The paper's appendix A.4 minimises a reach penalty at ratio ``0.65``, but
    that is an objective for the M4 grid search over *feasible* placements.  For
    the fixed-base MVP path the binding constraint is that the arm stays inside
    the ego camera's field of view *and* the workspace is dexterously reachable;
    the values here are that compromise (ADR-0009).
    """
    c = np.asarray(centroid_C_m, dtype=np.float64).reshape(3)
    horizontal = np.array([c[0], c[1], 0.0])
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-6:
        raise ValueError(
            "workspace centroid is directly above/below the camera; the base has no "
            "well-defined facing direction in the horizontal plane"
        )
    f = horizontal / norm
    distance = float(np.clip(reach_fraction, 0.05, 0.95)) * float(robot_reach_m)
    base = np.array(
        [c[0] - distance * f[0], c[1] - distance * f[1], c[2] - height_offset_m]
    )

    yaw = float(np.arctan2(f[1], f[0])) - np.pi / 2.0
    pitch = np.radians(float(pitch_deg))
    R = exp_so3(np.array([0.0, 0.0, yaw])) @ exp_so3(
        np.array([np.pi / 2.0 - pitch, 0.0, 0.0])
    )

    up_in_camera = R[:, 2]
    # allow a horizontal base z (a levelled base is legitimate); only a
    # *downward* pointing z is the bug this check exists to catch
    if up_in_camera[1] > 1e-9:
        raise ValueError(
            "base orientation would point the robot's +z downward in the camera frame "
            f"(component along camera -y is {-up_in_camera[1]:+.3f}), which would render "
            "the robot in front of the camera"
        )
    return make_transform(R, base)


def _default_base(lateral_m: float = 0.0, back_m: float = 0.0) -> np.ndarray:
    """Fallback base used when a case does not know its own workspace centroid.

    Kept only so :class:`CaseDefinition` always has a value; the case builder
    replaces it with :func:`base_for_workspace`, which needs the actual motion.
    """
    return base_for_workspace(
        np.array([0.06 + lateral_m, 0.10, 0.37 + back_m]), reach_fraction=0.26
    )


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------

_UPRIGHT = (0.0, 1.0, 0.0)      # jaw (grasp) axis, pointing from vf to thumb
_PALM_FWD = (0.0, 0.0, 1.0)     # wrist -> jaw midpoint


def _case1(frame: int) -> HandPose:
    """Right hand: pick at (0.01, -0.01, 0.28), move +x/+z, place, retreat."""
    n = 90
    t = frame / (n - 1)
    x = -0.015 + 0.075 * t
    y = 0.075 + 0.005 * t
    if t < 0.33:
        z, width = 0.36, 0.075
    elif t < 0.66:
        z = 0.36 - 0.02 * ((t - 0.33) / 0.33)
        width = 0.075 - 0.071 * ((t - 0.33) / 0.33)  # closes to 4 mm -> freeze kicks in
    elif t < 0.8:
        z, width = 0.34, 0.004
    else:
        # the retreat is deliberately shorter than the reach: the wrist sits
        # reach_m (~0.11 m) behind the jaw midpoint, so a large +z retreat would
        # push the wrist outside the 60 deg camera frustum and the overlay would
        # show joints leaving the frame
        z = 0.34 + 0.02 * ((t - 0.8) / 0.2)
        width = 0.004 + 0.071 * ((t - 0.8) / 0.2)
    return make_hand_pose(
        jaw_mid=(x, y, z),
        jaw_dir=_UPRIGHT,
        palm_dir=_PALM_FWD,
        handedness=1,
        gripper_width_m=width,
    )


def _case2(frame: int) -> HandPose:
    """The same motion with a left hand: the canonical frame must match."""
    n = 90
    t = frame / (n - 1)
    x = 0.075 - 0.10 * t
    y = 0.10 - 0.005 * t
    if t < 0.4:
        z, width = 0.38, 0.070
    elif t < 0.72:
        z = 0.38 - 0.03 * ((t - 0.4) / 0.32)
        width = 0.070 - 0.056 * ((t - 0.4) / 0.32)  # closes to 14 mm, stays valid
    else:
        z = 0.35 + 0.02 * ((t - 0.72) / 0.28)
        width = 0.014 + 0.056 * ((t - 0.72) / 0.28)
    return make_hand_pose(
        jaw_mid=(x, y, z),
        jaw_dir=_UPRIGHT,
        palm_dir=_PALM_FWD,
        handedness=-1,
        gripper_width_m=width,
    )


def _case3(frame: int) -> HandPose:
    """Stationary hand: zero motion is the only correct answer."""
    return make_hand_pose(
        jaw_mid=(0.025, 0.10, 0.37),
        jaw_dir=_UPRIGHT,
        palm_dir=_PALM_FWD,
        handedness=1,
        gripper_width_m=0.055,
    )


def _case4(frame: int) -> HandPose:
    """Stationary wrist, jaw line sweeping about the vertical axis.

    The sweep is 45 degrees rather than a full quarter turn because the Panda's
    wrist reaches its joint limits near 90 degrees of jaw rotation for this base
    placement, and an IK failure there would mask the property this case exists to
    check (rotation-vector direction and magnitude).  45 degrees is enough to pin
    the sign and the axis, and the motion is still a non-commutative rotation.
    """
    n = 60
    angle = np.pi / 4.0 * (frame / (n - 1))
    jaw = exp_so3(np.array([0.0, 1.0, 0.0]) * angle) @ np.asarray(_UPRIGHT, dtype=np.float64)
    palm = exp_so3(np.array([0.0, 1.0, 0.0]) * angle) @ np.asarray(_PALM_FWD, dtype=np.float64)
    return make_hand_pose(
        jaw_mid=(0.03, 0.10, 0.38),
        jaw_dir=jaw,
        palm_dir=palm,
        handedness=1,
        gripper_width_m=0.06,
    )


def _case5(frame: int) -> HandPose:
    """Degeneracies: closed jaw, velocity spikes, and a 14-frame dropout.

    The jaw closes to 4 mm and **stays** there for a stretch of frames.  That
    matters: the degeneracy rule freezes the orientation once the width drops
    below 10 mm, but the paper's Savitzky-Golay window is 21 frames, so a narrow
    dip would be smoothed away in the *raw* signal before the rule could fire.
    Holding the closure makes the freeze observable, which is the point of this
    case.

    The corruption itself (spikes at frames 10, 40, 70 and the dropout over
    frames 50-63) is applied by the fixture builder, so the analytic expectation
    stays exact for the frames that survive.
    """
    n = 80
    t = frame / (n - 1)
    x = 0.005 + 0.09 * t
    y = 0.10
    z = 0.37 - 0.02 * t
    if t < 0.30:
        width = 0.06
    elif t < 0.45:
        width = 0.06 - 0.056 * ((t - 0.30) / 0.15)  # closes to 4 mm
    elif t < 0.75:
        width = 0.004  # held below the 10 mm degeneracy threshold
    else:
        width = 0.004 + 0.056 * ((t - 0.75) / 0.25)
    return make_hand_pose(
        jaw_mid=(x, y, z),
        jaw_dir=_UPRIGHT,
        palm_dir=_PALM_FWD,
        handedness=1,
        gripper_width_m=width,
    )


def load_cases(path: str | Path | None = None) -> list[CaseDefinition]:
    """Return the five validation cases.

    ``configs/fixtures/cases.yaml`` carries the human-readable metadata
    (instruction, what each case checks); the motion generators live here so the
    ground truth stays code-reviewable and importable.
    """
    import yaml

    meta: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.is_file():
            with open(p, "r", encoding="utf-8") as fh:
                meta = yaml.safe_load(fh) or {}

    defaults = meta.get("defaults", {})
    width = int(defaults.get("width", 640))
    height = int(defaults.get("height", 480))
    fps = float(defaults.get("fps", 30.0))
    overrides: dict[str, dict[str, Any]] = {
        c["id"]: c for c in meta.get("cases", []) if isinstance(c, dict) and "id" in c
    }

    def _mk(
        case_id: str,
        sequence: Callable[[int], HandPose],
        *,
        num_frames: int,
        handedness: int,
        T_C_B: np.ndarray | None = None,
        corruptions: dict[str, Any] | None = None,
        object_xy_m: tuple[float, float] = (0.08, 0.10),
        reach_fraction: float = 0.40,
    ) -> CaseDefinition:
        o = overrides.get(case_id, {})
        if "T_C_B" in o:
            base = np.asarray(o["T_C_B"], dtype=np.float64)
        elif T_C_B is not None:
            base = np.asarray(T_C_B, dtype=np.float64)
        else:
            # place the base behind the workspace, using this case's own motion
            centroid = np.mean(
                [sequence(i).jaw_mid_C_m for i in range(num_frames)], axis=0
            )
            base = base_for_workspace(
                centroid, reach_fraction=float(o.get("reach_fraction", reach_fraction))
            )
        return CaseDefinition(
            case_id=case_id,
            instruction=str(o.get("instruction", case_id)),
            checks=str(o.get("checks", "")),
            num_frames=int(o.get("num_frames", num_frames)),
            fps=float(o.get("fps", fps)),
            handedness=handedness,
            width=int(o.get("width", width)),
            height=int(o.get("height", height)),
            sequence=sequence,
            T_C_B=base,
            corruptions=dict(corruptions or {}),
            object_xy_m=tuple(o.get("object_xy_m", object_xy_m)),  # type: ignore[arg-type]
            notes=list(o.get("notes", [])),
        )

    return [
        _mk("case1_right_translate", _case1, num_frames=90, handedness=1),
        _mk("case2_left_mirror", _case2, num_frames=90, handedness=-1),
        _mk("case3_static", _case3, num_frames=45, handedness=1),
        _mk("case4_rotation", _case4, num_frames=60, handedness=1),
        _mk(
            "case5_degenerate",
            _case5,
            num_frames=80,
            handedness=1,
            corruptions={
                "spike_frames": [10, 40, 70],
                "spike_position_m": 0.075,
                "dropout_span": [50, 64],
            },
        ),
    ]


def case_ground_truth(case: CaseDefinition) -> dict[str, Any]:
    """Per-frame analytic expectation for a case (used by the verifier)."""
    poses = [case.sequence(i) for i in range(case.num_frames)]
    kp = np.stack([p.keypoints_C_m for p in poses], axis=0).astype(np.float32)
    T = np.stack([expected_tcp_frame(p) for p in poses], axis=0).astype(np.float64)
    pos = np.stack([expected_tcp_position(p) for p in poses], axis=0)
    width = np.array([expected_gripper_width(p) for p in poses], dtype=np.float64)

    drops = case.corruptions.get("dropout_span")
    valid = np.ones(case.num_frames, dtype=bool)
    if drops:
        valid[drops[0] : drops[1]] = False

    deltas: dict[str, Any] = {}
    if case.num_frames > 1:
        deltas["delta_position_C_m"] = np.diff(pos, axis=0)
        deltas["delta_rotvec_C_rad"] = np.stack(
            [log_so3(T[i + 1, :3, :3] @ T[i, :3, :3].T) for i in range(case.num_frames - 1)],
            axis=0,
        )
    return {
        "case_id": case.case_id,
        "handedness": case.handedness,
        "num_frames": case.num_frames,
        "keypoints_C_m": kp,
        "T_C_TCP": T,
        "tcp_position_C_m": pos,
        "gripper_width_m": width,
        "valid": valid,
        "corruptions": dict(case.corruptions),
        **deltas,
    }


def _unused_helpers() -> None:  # pragma: no cover - keeps imports meaningful
    _ = (invert_transform, transform_points)
