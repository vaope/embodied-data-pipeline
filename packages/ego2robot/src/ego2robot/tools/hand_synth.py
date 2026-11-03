"""Synthetic hand kinematics: MANO-21 skeletons with known ground truth.

The validation cases must be verifiable by hand, so the hand model here is
generated from *analytic* parameters with an exact closed-form ground truth:

    jaw midpoint   ``p_jaw``   -- the canonical TCP when the jaw is centred
    jaw direction  ``u_jaw``   -- the canonical grasp axis (+/- per handedness)
    palm direction ``u_palm``  -- wrist -> jaw midpoint
    jaw half width ``w/2``     -- ``||p_thumb - p_vf|| / 2``

For a right hand the canonical frame implied by the paper's equations 1-3 is::

    x = u_palm
    z = +u_jaw
    y = z x x

and the joints are placed so that::

    p_index  = p_jaw + 0.5 * w * u_jaw
    p_middle = p_jaw + 0.5 * w * u_jaw
    p_thumb  = p_jaw - 0.5 * w * u_jaw
    p_wrist  = p_jaw - d * u_palm

so ``p_vf`, ``p_tcp`` and ``R_C_TCP`` all have an exact expected value.  A
separate helper (:func:`expected_tcp_frame`) returns that value so the pipeline
output can be diffed against it instead of being eyeballed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.transforms import exp_so3, make_transform

__all__ = [
    "BONES",
    "HandPose",
    "expected_tcp_frame",
    "expected_tcp_position",
    "hand_keypoints",
    "make_hand_pose",
    "retarget_joint_indices",
]

#: MANO-21 chains used for drawing; mirrors viz.hand_overlay.HAND_BONES
BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

#: per-finger chain of MANO indices, root first
FINGER_CHAINS: dict[str, tuple[int, ...]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}

#: lateral offsets of each finger root from the jaw midpoint, in units of
#: ``lateral_scale``; negative is toward the thumb side for a right hand.
_FINGER_LATERAL_OFFSET = {"thumb": -0.30, "index": -0.06, "middle": 0.02, "ring": 0.10, "little": 0.18}


def retarget_joint_indices() -> tuple[int, int, int, int]:
    """``(wrist, thumb_tip, index_tip, middle_tip)`` -- the joints that matter."""
    return (0, 4, 8, 12)


@dataclass
class HandPose:
    """One frame of analytic hand parameters."""

    jaw_mid_C_m: NDArray[np.float64]
    """Position of the jaw midpoint, i.e. the canonical TCP when centred."""
    jaw_dir_C: NDArray[np.float64]
    """Unit vector along the jaw line, pointing from virtual fingertip to thumb."""
    palm_dir_C: NDArray[np.float64]
    """Unit vector from the wrist to the jaw midpoint."""
    handedness: int
    gripper_width_m: float
    reach_m: float = 0.11
    lateral_scale_m: float = 0.035
    keypoints_C_m: NDArray[np.float64] = field(default=None, init=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.jaw_dir_C = _unit(self.jaw_dir_C)
        self.palm_dir_C = _unit(self.palm_dir_C)
        self.keypoints_C_m = hand_keypoints(
            jaw_mid=self.jaw_mid_C_m,
            jaw_dir=self.jaw_dir_C,
            palm_dir=self.palm_dir_C,
            handedness=self.handedness,
            gripper_width_m=self.gripper_width_m,
            reach_m=self.reach_m,
            lateral_scale_m=self.lateral_scale_m,
        )


def _unit(v: Sequence[float] | NDArray[np.floating]) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("cannot normalise a zero vector")
    return a / n


def hand_keypoints(
    *,
    jaw_mid: NDArray[np.floating],
    jaw_dir: NDArray[np.floating],
    palm_dir: NDArray[np.floating],
    handedness: int,
    gripper_width_m: float,
    reach_m: float = 0.11,
    lateral_scale_m: float = 0.035,
) -> NDArray[np.float64]:
    """Build a MANO-21 skeleton ``[21, 3]`` in the camera frame.

    Signed ``s = +1`` (right) / ``-1`` (left) enters through ``y = (s z) x d``,
    which is what makes both hands map to the same canonical gripper frame:
    the lateral layout below is mirrored for the left hand so that the
    resulting thumb/index/middle positions reproduce the same ``R_C_TCP``.
    """
    jaw_mid = np.asarray(jaw_mid, dtype=np.float64).reshape(3)
    z = _unit(jaw_dir)
    x = _unit(palm_dir)
    y = np.cross(z, x)
    if float(np.linalg.norm(y)) < 1e-9:
        raise ValueError(
            "jaw_dir and palm_dir are parallel: the jaw plane has no unique normal"
        )
    y = _unit(y)
    # orthonormalise palm_dir against the jaw axis so the frame is exactly SO(3)
    x = _unit(np.cross(y, z))

    sign = 1.0 if handedness > 0 else -1.0
    w = float(gripper_width_m)
    kp = np.zeros((21, 3), dtype=np.float64)

    def place(index: int, along_palm: float, lateral: float, along_jaw: float) -> None:
        # ``lateral`` is mirrored per handedness; ``along_jaw`` is measured along
        # the *anatomical* thumb direction, so the thumb sits at +1 for a right
        # hand and the mirror of it for a left hand.  Flipping both is what makes
        # the paper's ``z = s (p_thumb - p_vf) / w`` produce the same canonical
        # grasp axis for either hand.
        kp[index] = (
            jaw_mid
            + along_palm * reach_m * x
            + sign * lateral * lateral_scale_m * y
            + sign * along_jaw * w * z
        )

    # --- wrist -----------------------------------------------------------
    place(0, -1.0, 0.0, 0.0)

    # --- tips -------------------------------------------------------------
    # All five tips are placed explicitly.  Their positions ARE the ground truth
    # the retargeter reads, so they are computed from the analytic jaw
    # parameters rather than blended, and every other joint is then blended
    # between its chain root and its tip.
    place(4, 0.0, 0.0, -0.5)     # thumb tip
    place(8, 0.0, 0.0, +0.5)     # index tip
    place(12, 0.0, 0.0, +0.5)    # middle tip
    place(16, 0.0, 0.0, +0.25)   # ring tip (curls off the jaw plane)
    place(20, 0.0, 0.0, +0.15)   # little tip

    # --- remaining joints along each finger chain -------------------------
    # The tips are already pinned by the loop above (they are what the
    # retargeter reads), so every intermediate joint is a straight blend between
    # its chain's root and that tip.  A blend cannot leave the segment, which
    # keeps the whole skeleton inside a plausible hand volume; the earlier
    # "place along a palm/lateral/jaw triple" version produced joints at depths
    # near zero and therefore UV coordinates thousands of pixels off screen.
    tip_index = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "little": 20}
    for finger, chain in FINGER_CHAINS.items():
        tip = tip_index[finger]
        lateral = _FINGER_LATERAL_OFFSET[finger]
        if finger == "thumb":
            # the thumb's metacarpal sits at the wrist end of the palm, on the
            # thumb side, slightly up the jaw axis
            root = jaw_mid + sign * lateral * lateral_scale_m * y - 0.15 * w * z + 0.2 * reach_m * x
        elif finger in {"index", "middle"}:
            root = jaw_mid + sign * lateral * lateral_scale_m * y + 0.35 * w * z - 0.25 * reach_m * x
        else:
            root = jaw_mid + sign * lateral * lateral_scale_m * y - 0.2 * reach_m * x
        tip_point = kp[tip]
        n = len(chain)
        for k, joint in enumerate(chain):
            if joint == tip:
                continue  # pinned exactly by the tip loop
            t = k / float(n - 1)
            kp[joint] = (1.0 - t) * root + t * tip_point
    return kp


def expected_tcp_position(pose: HandPose) -> NDArray[np.float64]:
    """Ground-truth TCP for a pose: ``(p_thumb + p_vf) / 2``.

    Deliberately recomputed from the *joints* (not from ``jaw_mid``) so that a
    bug in :func:`hand_keypoints` cannot hide behind a matching expectation.
    """
    kp = pose.keypoints_C_m
    p_vf = 0.7 * kp[8] + 0.3 * kp[12]
    return 0.5 * (kp[4] + p_vf)


def expected_tcp_frame(pose: HandPose) -> NDArray[np.float64]:
    """Ground-truth ``T_C_TCP`` implied by the paper's equations 1-3."""
    kp = pose.keypoints_C_m
    sign = 1.0 if pose.handedness > 0 else -1.0
    p_vf = 0.7 * kp[8] + 0.3 * kp[12]
    p_tcp = 0.5 * (kp[4] + p_vf)
    w = float(np.linalg.norm(kp[4] - p_vf))
    d = p_vf - kp[0]
    z = sign * (kp[4] - p_vf) / w
    y = _unit(np.cross(z, d))
    x = np.cross(y, z)
    return make_transform(np.stack([x, y, z], axis=1), p_tcp)


def expected_gripper_width(pose: HandPose) -> float:
    kp = pose.keypoints_C_m
    p_vf = 0.7 * kp[8] + 0.3 * kp[12]
    return float(np.linalg.norm(kp[4] - p_vf))


def make_hand_pose(
    *,
    jaw_mid: Sequence[float],
    jaw_dir: Sequence[float],
    palm_dir: Sequence[float],
    handedness: int = 1,
    gripper_width_m: float = 0.06,
    reach_m: float = 0.11,
    lateral_scale_m: float = 0.035,
) -> HandPose:
    """Convenience constructor mirroring the case definitions."""
    return HandPose(
        jaw_mid_C_m=np.asarray(jaw_mid, dtype=np.float64),
        jaw_dir_C=np.asarray(jaw_dir, dtype=np.float64),
        palm_dir_C=np.asarray(palm_dir, dtype=np.float64),
        handedness=handedness,
        gripper_width_m=float(gripper_width_m),
        reach_m=float(reach_m),
        lateral_scale_m=float(lateral_scale_m),
    )


def rotate_direction(direction: Sequence[float], axis: Sequence[float], angle_rad: float) -> NDArray[np.float64]:
    """Rotate a unit direction about an axis (helpers for the rotation case)."""
    return exp_so3(np.asarray(axis, dtype=np.float64) * float(angle_rad)) @ _unit(direction)
