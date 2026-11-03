"""Joint two-arm render: RGB, ownership map, per-arm masks and metric depth.

ADR-0022 decision 9 / PLAN-02 B3: the two arms are rendered **once, in one scene**,
so the depth order between them is physical (from MuJoCo's depth buffer) instead of a
draw-order guess.  Outputs per frame:

``rgb``
    robot-only RGB (no human, no background) — the compositing input;
``owner``
    ``uint8`` H×W: 0 background, 1 right arm, 2 left arm (from segmentation IDs);
``arm_mask`` / ``gripper_mask``
    per-arm arm-body and gripper masks, derived from the same segmentation pass;
``depth_m``
    metric depth in the **camera** frame, NaN where nothing was hit.

The camera is placed exactly like the single-arm renderer (``T_W_C`` = inverse
``T_C_B`` of the right arm, times the MuJoCo-camera-to-optical basis), because the
generated scene's world is the right arm's base frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.schema import CameraModel
from ..core.transforms import invert_transform, quat_xyzw_from_matrix
from .render import MJ_CAM_TO_OPTICAL_BASIS

__all__ = ["RigRenderFrame", "RigRenderer"]

GRIPPER_BODIES = {"hand", "left_finger", "right_finger"}


@dataclass
class RigRenderFrame:
    frame_index: int
    rgb: NDArray[np.uint8]
    owner: NDArray[np.uint8]
    arm_mask: NDArray[np.bool_]
    gripper_mask: NDArray[np.bool_]
    depth_m: NDArray[np.float32]
    per_arm_pixels: dict[str, int]


class RigRenderer:
    """Offscreen renderer over the generated two-arm scene."""

    def __init__(
        self,
        scene_path: str | Path,
        camera: CameraModel,
        T_C_B_right: NDArray[np.float64],
        *,
        camera_name: str = "rig_cam",
        shadows: bool = True,
        max_geom: int = 40000,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(scene_path)))
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, camera.width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, camera.height)
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(
            self.model, height=camera.height, width=camera.width, max_geom=max_geom
        )
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self._cam_id < 0:
            raise RuntimeError(f"generated rig scene has no camera named {camera_name!r}")

        # same convention as vision/render.py::set_camera_pose
        T_B_C = invert_transform(np.asarray(T_C_B_right, dtype=np.float64))
        R_W_cam = T_B_C[:3, :3] @ MJ_CAM_TO_OPTICAL_BASIS
        q = quat_xyzw_from_matrix(R_W_cam)
        self.model.cam_pos[self._cam_id] = T_B_C[:3, 3]
        self.model.cam_quat[self._cam_id] = np.array([q[3], q[0], q[1], q[2]])
        self.model.cam_fovy[self._cam_id] = float(
            2.0 * np.degrees(np.arctan(0.5 * camera.height / camera.fy))
        )
        self.model.cam_resolution[self._cam_id] = [camera.width, camera.height]

        self._shadows = shadows
        self._owner: dict[int, int] = {}
        self._gripper: dict[int, bool] = {}
        for gid in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[gid])
            name = self._body_name(body_id)
            if not name:
                self._owner[gid] = 0
                continue
            prefix = name[0] if name[0] in {"R", "L"} else ""
            suffix = name[1:] if prefix else name
            self._owner[gid] = 1 if prefix == "R" else 2 if prefix == "L" else 0
            self._gripper[gid] = suffix in GRIPPER_BODIES

    # -- helpers -----------------------------------------------------------

    def _body_name(self, body_id: int) -> str:
        return self._mujoco.mj_id2name(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_id
        ) or ""

    def _joint_id(self, name: str) -> int:
        jid = int(self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            raise KeyError(f"{name!r} is not in the rig scene")
        return jid

    def set_arm(
        self,
        prefix: str,
        joint_names: Sequence[str],
        qpos: NDArray[np.float64],
        gripper_names: Sequence[str] | None = None,
        gripper_qpos: NDArray[np.float64] | None = None,
    ) -> None:
        for name, value in zip(joint_names, np.asarray(qpos, dtype=np.float64)):
            self.data.qpos[int(self.model.jnt_qposadr[self._joint_id(prefix + name)])] = value
        if gripper_names and gripper_qpos is not None:
            for name, value in zip(gripper_names, np.asarray(gripper_qpos, dtype=np.float64)):
                self.data.qpos[int(self.model.jnt_qposadr[self._joint_id(prefix + name)])] = value

    def render(self, frame_index: int = 0) -> RigRenderFrame:
        mujoco = self._mujoco
        mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera="rig_cam")

        self._renderer.enable_segmentation_rendering()
        seg = np.asarray(self._renderer.render())[:, :, 0].astype(np.int32)
        self._renderer.disable_segmentation_rendering()

        self._renderer.enable_depth_rendering()
        depth = np.asarray(self._renderer.render(), dtype=np.float32)
        self._renderer.disable_depth_rendering()
        depth = depth[:, :, 0] if depth.ndim == 3 else depth

        rgb = np.asarray(self._renderer.render(), dtype=np.uint8)

        owner = np.zeros(seg.shape, dtype=np.uint8)
        gripper = np.zeros(seg.shape, dtype=bool)
        for gid, side in self._owner.items():
            if side == 0:
                continue
            mask = seg == gid
            if mask.any():
                owner[mask] = side
                if self._gripper.get(gid, False):
                    gripper |= mask
        arm = owner > 0
        return RigRenderFrame(
            frame_index=frame_index,
            rgb=rgb,
            owner=owner,
            arm_mask=arm,
            gripper_mask=gripper,
            depth_m=np.where(arm | gripper, depth, np.nan).astype(np.float32),
            per_arm_pixels={
                "right": int(np.count_nonzero(owner == 1)),
                "left": int(np.count_nonzero(owner == 2)),
            },
        )

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:  # noqa: BLE001 - closing a renderer must not mask results
            pass

    def __enter__(self) -> "RigRenderer":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False
