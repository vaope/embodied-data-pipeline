"""Depth-aware robot compositing (paper appendix A.4, system-design 9.6).

The rule the paper states is

.. code-block:: text

    I_final(u,v) = I_robot(u,v)   if D_robot(u,v) < D_scene(u,v) and M_robot(u,v) = 1
                 = I_inpaint(u,v) otherwise

with one asymmetry that matters as much as the inequality: the **arm body** is not
depth tested at all.  The arm stands between the camera and the workspace, so a
scene-depth sample on an arm pixel belongs to whatever is behind the arm; testing it
would delete the forearm wherever it crosses a table edge.  Only the **gripper** is
compared, because only the gripper enters the scene and gets occluded by it.

The second asymmetry is the hand-mask exemption.  The hand region is exactly what
inpainting replaced, so a monocular depth estimate inside it describes the surface
*behind* the hand.  Comparing the gripper against that would hide the jaws at the
grasp point — precisely where the composite must be right.  Gripper pixels inside the
dilated hand mask are therefore kept, and counted separately so the exemption stays
visible instead of becoming a silent licence.

Nothing here decides whether the inputs are trustworthy: the caller must pass scene
depth that already passed :meth:`ego2robot.vision.scene_depth.SceneDepth.assert_composable`,
and :func:`composite_frame` re-checks the mode itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .scene_depth import DepthMode, DepthModeError, _COMPOSABLE

__all__ = [
    "DECISION_ARM",
    "DECISION_BACKGROUND",
    "DECISION_GRIPPER_EXEMPT",
    "DECISION_GRIPPER_OCCLUDED",
    "DECISION_GRIPPER_VISIBLE",
    "CompositeFrame",
    "composite_frame",
    "decision_legend",
]

#: Decision-map codes.  Small, stable integers so a reviewer can read the PNG as an
#: indexed image and a script can count classes without guessing colours.
DECISION_BACKGROUND = 0
DECISION_ARM = 1
DECISION_GRIPPER_VISIBLE = 2
DECISION_GRIPPER_OCCLUDED = 3
DECISION_GRIPPER_EXEMPT = 4

#: Fixed palette for the decision map; index == code.
_PALETTE = np.array(
    [
        [16, 18, 22],      # 0 background
        [90, 200, 255],    # 1 arm, always visible
        [60, 235, 120],    # 2 gripper visible (in front of the scene)
        [255, 70, 70],     # 3 gripper occluded by the scene
        [255, 190, 60],    # 4 gripper kept by the hand-mask exemption
    ],
    dtype=np.uint8,
)


def decision_legend() -> dict[int, str]:
    """Human-readable decision codes for manifests and review pages."""
    return {
        DECISION_BACKGROUND: "background",
        DECISION_ARM: "arm_always_visible",
        DECISION_GRIPPER_VISIBLE: "gripper_in_front",
        DECISION_GRIPPER_OCCLUDED: "gripper_occluded_by_scene",
        DECISION_GRIPPER_EXEMPT: "gripper_exempt_hand_mask",
    }


def decision_rgb(decision: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Colourise a decision map."""
    return _PALETTE[np.clip(decision, 0, len(_PALETTE) - 1)]


@dataclass
class CompositeFrame:
    """One composited frame plus the bookkeeping that explains it."""

    image: NDArray[np.uint8]
    decision: NDArray[np.uint8]
    visible_mask: NDArray[np.bool_]
    occluded_mask: NDArray[np.bool_]
    metrics: dict[str, Any] = field(default_factory=dict)

    def decision_rgb(self) -> NDArray[np.uint8]:
        return decision_rgb(self.decision)


def _dilate(mask: NDArray[np.bool_], kernel: int) -> NDArray[np.bool_]:
    if kernel <= 1:
        return mask.copy()
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, structure=np.ones((kernel, kernel), dtype=bool))


def composite_frame(
    background: NDArray[np.uint8],
    robot_rgb: NDArray[np.uint8],
    robot_mask: NDArray[np.bool_],
    gripper_mask: NDArray[np.bool_],
    robot_depth_m: NDArray[np.float32],
    scene_depth_m: NDArray[np.float32],
    *,
    scene_mode: DepthMode,
    hand_mask: NDArray[np.bool_] | None = None,
    mode: Literal["depth_aware", "debug_alpha"] = "depth_aware",
    hand_mask_dilation_kernel: int = 5,
    arm_body_always_visible: bool = True,
    occlusion_bias_m: float = 0.0,
) -> CompositeFrame:
    """Composite one frame.

    ``mode='debug_alpha'`` draws the robot wherever it has a mask and never depth
    tests; it is the documented debug path (ADR-0008) and its caller must write it
    under a different name.
    """
    bg = np.asarray(background, dtype=np.uint8)
    rgb = np.asarray(robot_rgb, dtype=np.uint8)
    if bg.shape != rgb.shape:
        raise ValueError(f"background {bg.shape} and robot RGB {rgb.shape} differ")
    h, w = bg.shape[:2]
    for name, mask in (("robot_mask", robot_mask), ("gripper_mask", gripper_mask)):
        if mask.shape != (h, w):
            raise ValueError(f"{name} {mask.shape} does not match the image {(h, w)}")
    if robot_depth_m.shape != (h, w) or scene_depth_m.shape != (h, w):
        raise ValueError("depth arrays must match the image shape")

    arm = np.asarray(robot_mask, dtype=bool) & ~np.asarray(gripper_mask, dtype=bool)
    gripper = np.asarray(gripper_mask, dtype=bool)
    if mode == "depth_aware" and scene_mode not in _COMPOSABLE:
        raise DepthModeError(
            f"composite_frame refuses scene depth with mode={scene_mode!r}; "
            "use mode='debug_alpha' for a non-metric preview (ADR-0008)"
        )

    decision = np.zeros((h, w), dtype=np.uint8)
    image = bg.copy()

    visible_arm = arm
    if not arm_body_always_visible:
        # Kept for completeness; the paper and the design both require True.
        visible_arm = arm & (
            ~np.isfinite(scene_depth_m) | (robot_depth_m < scene_depth_m)
        )

    d_robot = np.asarray(robot_depth_m, dtype=np.float32)
    d_scene = np.asarray(scene_depth_m, dtype=np.float32)
    depth_known = np.isfinite(d_scene) & (d_scene > 0)
    robot_known = np.isfinite(d_robot) & (d_robot > 0)

    exempt = np.zeros((h, w), dtype=bool)
    if hand_mask is not None and hand_mask_dilation_kernel > 0:
        exempt = _dilate(np.asarray(hand_mask, dtype=bool), hand_mask_dilation_kernel)

    if mode == "debug_alpha":
        occluded_gripper = np.zeros_like(gripper)
        exempt_gripper = np.zeros_like(gripper)
        scene_invalid_gripper = np.zeros_like(gripper)
    else:
        behind = depth_known & robot_known & (
            (d_robot + np.float32(occlusion_bias_m)) >= d_scene
        )
        # Conservative policy: an unknown scene depth cannot hide the robot.  The
        # alternative (treat unknown as "in front") would delete the gripper wherever
        # the depth model has a hole, which is the failure mode a reviewer would
        # notice last and trust most.
        exempt_gripper = gripper & exempt & behind
        occluded_gripper = behind & gripper & ~exempt
        scene_invalid_gripper = gripper & ~depth_known & ~exempt

    if visible_arm.any():
        image[visible_arm] = rgb[visible_arm]
        decision[visible_arm] = DECISION_ARM
    if mode == "debug_alpha":
        if gripper.any():
            image[gripper] = rgb[gripper]
            decision[gripper] = DECISION_GRIPPER_VISIBLE
    else:
        if exempt_gripper.any():
            image[exempt_gripper] = rgb[exempt_gripper]
            decision[exempt_gripper] = DECISION_GRIPPER_EXEMPT
        plain_visible = gripper & ~occluded_gripper & ~exempt_gripper
        if plain_visible.any():
            image[plain_visible] = rgb[plain_visible]
            decision[plain_visible] = DECISION_GRIPPER_VISIBLE
        if occluded_gripper.any():
            decision[occluded_gripper] = DECISION_GRIPPER_OCCLUDED

    occluded_mask = occluded_gripper.copy()
    visible_mask = (decision == DECISION_ARM) | (decision == DECISION_GRIPPER_VISIBLE) | (
        decision == DECISION_GRIPPER_EXEMPT
    )
    metrics = {
        "arm_pixels": int(arm.sum()),
        "gripper_pixels": int(gripper.sum()),
        "arm_visible_pixels": int((decision == DECISION_ARM).sum()),
        "gripper_visible_pixels": int(
            (decision == DECISION_GRIPPER_VISIBLE).sum() + (decision == DECISION_GRIPPER_EXEMPT).sum()
        ),
        "gripper_occluded_pixels": int(occluded_gripper.sum()),
        "gripper_exempt_pixels": int(exempt_gripper.sum()),
        "gripper_scene_depth_unknown_pixels": int(scene_invalid_gripper.sum()),
        "robot_visible_pixels": int(visible_mask.sum()),
        "hand_mask_pixels": int(np.asarray(hand_mask, dtype=bool).sum())
        if hand_mask is not None
        else 0,
        "scene_depth_valid_fraction": round(float(depth_known.mean()), 6),
        "mode": mode,
        "occlusion_bias_m": float(occlusion_bias_m),
    }
    return CompositeFrame(
        image=image,
        decision=decision,
        visible_mask=visible_mask,
        occluded_mask=occluded_mask,
        metrics=metrics,
    )
