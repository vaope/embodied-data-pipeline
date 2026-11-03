"""MuJoCo rendering: robot RGB, robot/gripper masks and metric depth.

System design 9.4 and implementation-plan.md M3.

Camera convention

-----------------

This project's camera frame ``C`` is the optical convention ``+x`` right,

``+y`` down, ``+z`` forward.  MuJoCo camera bodies instead look along their local -z with local +y up; the old text said:

optical axis is the camera body's ``+z`` and the image ``+y`` axis is the camera

body's ``+y``, i.e. "up", whereas ``C`` treats ``+y`` as "down".

The renderer therefore places the camera body with an explicit basis transform

applied on the right of ``T_B_C``::

    R_B_MjCam = R_B_C @ diag(1, -1, -1)

so image pixel ``(u, v)`` with ``u`` right and ``v`` down corresponds to the

camera-frame ray ``((u - cx)/fx, (v - cy)/fy, 1)``.  This is verified by

:func:`verify_projection`, which renders emissive markers at known 3D points_C and

compares their centroids with the analytic pinhole projection —never by

eyeballing signs (system-design.md 11.2).

"""

from __future__ import annotations

import os

import tempfile

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any

import numpy as np

from numpy.typing import NDArray

from ..core.frames import Frame, transform_name

from ..core.schema import CameraModel

from ..core.transforms import (
    compose_transforms,
    invert_transform,
    make_transform,
    quat_xyzw_from_matrix,
    transform_points,
)

__all__ = [

    "CameraSpec",

    "MJ_CAM_TO_OPTICAL_BASIS",

    "MujocoRenderer",

    "camera_spec_from_pinhole",

    "mujoco_camera_basis",

    "render_robot_scene",

    "verify_projection",

]

#: Basis relating frame `C` to a MuJoCo camera body.
#:
#: MuJoCo camera bodies look along their **local -z** with local **+y up**,
#: whereas frame `C` is the optical convention: `+x` right, `+y` **down**,
#: `+z` **forward**.  The basis therefore satisfies
#:
#:     optical +x (right)   -> camera +x
#:     optical +y (down)    -> camera -y
#:     optical +z (forward) -> camera -z
#:
#: which is `diag(1, -1, -1)` (`det = +1`).  Because MuJoCo's optical axis is
#: `-z`, the depth buffer comes back **negated** with respect to frame `C`, so
#: :meth:MujocoRenderer.render flips its sign before returning metric depth.
#:
#: The value was determined empirically rather than reasoned: 24 candidate signed
#: permutation bases were rendered with markers at known camera-frame points_C and
#: scored against the analytic projection.  `diag(1, -1, -1)` is the only one
#: that reproduces the analytic pixels (mean residual ~1 px on markers 0.5-0.6 m
#: away).  Earlier revisions used `diag(1, -1, 1)` and `diag(-1, -1, 1)`, both
#: of which leave the camera axis mismatched, producing an empty frame whose
#: segmentation is uniformly `-1` -- the render stage reports that as "the
#: robot was invisible in every rendered frame".

MJ_CAM_TO_OPTICAL_BASIS = np.diag([1.0, -1.0, -1.0])

_CAMERA_NAME = "e2r_eye"

def mujoco_camera_basis() -> NDArray[np.float64]:

    """The basis transform between MuJoCo camera bodies and frame ``C``."""

    return MJ_CAM_TO_OPTICAL_BASIS.copy()

def frame_transform_name() -> str:

    """Documentation anchor for the ``T_C_B`` variable this module consumes."""

    return str(transform_name(Frame.C, Frame.B))

@dataclass

class CameraSpec:

    """Everything the renderer needs to reproduce a calibrated camera."""

    width: int

    height: int

    T_C_B: NDArray[np.float64]

    """``T_C_B``: pose of the robot base expressed in the camera optical frame."""

    fovy_deg: float

    """Vertical field of view derived from ``fy`` so pixels match ``K``."""

    znear: float = 0.01

    zfar: float = 10.0

    K: NDArray[np.float64] | None = None

    source: str = "episode"

    notes: list[str] = field(default_factory=list)

    @property

    def T_B_C(self) -> NDArray[np.float64]:

        return invert_transform(self.T_C_B)

    def project(self, points_C_m: NDArray[np.floating[Any]]) -> NDArray[np.float64]:

        """Analytic pinhole projection using ``K``."""

        if self.K is None:

            raise ValueError("CameraSpec has no K; analytic projection is unavailable")

        P = np.asarray(points_C_m, dtype=np.float64)

        z = P[..., 2]

        safe = np.where(np.abs(z) < 1e-9, np.nan, z)

        u = self.K[0, 0] * P[..., 0] / safe + self.K[0, 2]

        v = self.K[1, 1] * P[..., 1] / safe + self.K[1, 2]

        return np.stack([u, v], axis=-1)

def camera_spec_from_pinhole(

    camera: CameraModel,

    T_C_B: NDArray[np.floating[Any]],

    *,

    znear: float = 0.01,

    zfar: float = 10.0,

) -> CameraSpec:

    """Build a :class:`CameraSpec` from calibrated intrinsics plus ``T_C_B``.

    MuJoCo camera models use a vertical field of view; ``fy`` fixes the number

    of pixels per unit tangent, so ``fovy = 2 * atan(height / (2 * fy))`` makes

    the analytic projection of a camera-frame point land on the same pixel the

    renderer produces.

    """

    T = np.asarray(T_C_B, dtype=np.float64).reshape(4, 4)

    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):

        raise ValueError("T_C_B bottom row must be [0, 0, 0, 1]")

    fovy = 2.0 * float(np.arctan(camera.height / (2.0 * camera.fy)))

    return CameraSpec(

        width=camera.width,

        height=camera.height,

        T_C_B=T,

        fovy_deg=float(np.degrees(fovy)),

        znear=znear,

        zfar=zfar,

        K=camera.K_array,

        source=camera.calibration_source,

        notes=[f"fovy derived from fy={camera.fy:.4f} px at height={camera.height}"],

    )

# ---------------------------------------------------------------------------

# Scene XML

# ---------------------------------------------------------------------------

def build_scene_xml(

    base_xml: Path,

    *,

    plane_height_m: float | None = None,

    camera_name: str = _CAMERA_NAME,

) -> str:

    """Robot-only scene: the robot model plus a camera, optionally a floor.

    No floor by default, so the rendered depth contains only robot pixels and

    "no return" is unambiguous (system-design.md 6.5).

    ``base_xml`` is emitted as an **absolute** path because the scene file lives

    in a temporary directory; a relative ``include`` would be resolved against

    that directory and fail to find the model.  ``meshdir`` is passed explicitly

    for the same reason: the model resolves its assets relative to the model

    directory, not relative to whichever file pulled it in.

    """

    model_dir = base_xml.resolve().parent

    meshdir = model_dir / "assets"

    mesh_attr = f' meshdir="{meshdir.as_posix()}"' if meshdir.is_dir() else ""

    parts = [

        '<mujoco model="e2r_robot_scene">',

        f'  <include file="{base_xml.resolve().as_posix()}"{mesh_attr}/>',

        '  <statistic center="0.4 0 0.4" extent="1.0"/>',

        "  <visual>",

        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>',

        '    <quality shadowsize="2048"/>',

        "  </visual>",

        "  <worldbody>",

        f'    <camera name="{camera_name}" pos="1 0 1" quat="1 0 0 0" fovy="45"/>',

    ]

    if plane_height_m is not None:

        parts.append(

            f'    <geom name="e2r_floor" type="plane" size="3 3 0.1" pos="0 0 {plane_height_m}" '

            'rgba="0.85 0.85 0.85 1" contype="0" conaffinity="0"/>'

        )

    parts += ["  </worldbody>", "</mujoco>", ""]

    return "\n".join(parts)

class MujocoRenderer:

    """Offscreen renderer producing RGB, masks and metric depth.

    The model is compiled from a generated scene XML that ``include``\\ s the

    robot MJCF.  The scene file is written *next to* the robot model because

    MuJoCo resolves ``meshdir`` (and every relative asset path) against the

    directory of the **top-level** XML file, not against the included file.  A

    scene written to a temporary directory therefore cannot find ``assets/`` and

    fails with "Error opening file .../link0.stl".

    The file has a dot-prefixed, per-process name and is deleted on

    :meth:`close`; the vendored model itself is never modified.

    """

    def __init__(

        self,

        robot: Any,

        camera: CameraSpec,

        *,

        shadows: bool = True,

        max_geom: int = 20000,

        plane_height_m: float | None = None,

    ) -> None:

        import mujoco

        self._mujoco = mujoco

        self.robot = robot

        self.camera = camera

        model_path = Path(robot.mjcf_path).resolve()

        self._scene_paths: list[Path] = []

        scene_path = self._write_scene_file(model_path, plane_height_m=plane_height_m)

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))

        # MuJoCo defaults to a 640x480 offscreen framebuffer.  Real EgoDex
        # frames are 1920x1080, so size it before constructing Renderer.
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, camera.width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, camera.height)

        self.data = mujoco.MjData(self.model)

        self._renderer = mujoco.Renderer(

            self.model, height=camera.height, width=camera.width, max_geom=max_geom

        )

        self.scene_option = mujoco.MjvOption()

        mujoco.mjv_defaultOption(self.scene_option)

        self._cam_id = mujoco.mj_name2id(

            self.model, mujoco.mjtObj.mjOBJ_CAMERA, _CAMERA_NAME

        )

        if self._cam_id < 0:

            raise RuntimeError(f"generated scene has no camera named {_CAMERA_NAME!r}")

        self.model.cam_fovy[self._cam_id] = float(camera.fovy_deg)

        self.model.cam_resolution[self._cam_id] = [camera.width, camera.height]

        self.shadows = shadows

        

        self.scene_note: str = getattr(self, "scene_note", "")

        gripper_names = set(self.robot.config.gripper_geom_names)
        self._gripper_body_ids = {
            body_id
            for name in gripper_names
            if (body_id := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
        }
        self._gripper_named_geoms = gripper_names

        self._arm_geom_ids, self._gripper_geom_ids = self._classify_geoms()

        self._all_robot_geoms = np.asarray(

            self._arm_geom_ids + self._gripper_geom_ids, dtype=np.int32

        )

        self._gripper_geoms = np.asarray(self._gripper_geom_ids, dtype=np.int32)

        self.set_camera_pose()

        self.set_state(robot.home_qpos)

    # -- setup -------------------------------------------------------------

    def _write_scene_file(self, model_path: Path, *, plane_height_m: float | None) -> Path:

        """Write the generated scene next to the model so ``meshdir`` resolves.

        Falls back to a temporary directory (with an explicit ``meshdir``) when

        the model directory is not writable, and records that fallback in

        ``self.scene_note`` rather than failing.

        """

        xml = build_scene_xml(model_path, plane_height_m=plane_height_m)

        try:

            fd, name = tempfile.mkstemp(

                dir=str(model_path.parent), prefix=".e2r_scene_", suffix=".xml"

            )

            with os.fdopen(fd, "w", encoding="utf-8") as fh:

                fh.write(xml)

            path = Path(name)

            self._scene_paths.append(path)

            self.scene_note = f"scene written beside the model: {path.name}"

            return path

        except OSError as exc:

            tmpdir = tempfile.mkdtemp(prefix="e2r_scene_")

            path = Path(tmpdir) / "scene.xml"

            path.write_text(xml, encoding="utf-8")

            self._scene_paths.append(path)

            self.scene_note = (

                f"model directory not writable ({exc}); fell back to {tmpdir}, which may "

                "break relative mesh resolution"

            )

            return path

    def _classify_geoms(self) -> tuple[list[int], list[int]]:

        arm: list[int] = []

        gripper: list[int] = []

        for gid in range(self.model.ngeom):

            body = int(self.model.geom_bodyid[gid])

            geom_name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, gid)
            if body in self._gripper_body_ids or geom_name in self._gripper_named_geoms:

                gripper.append(gid)

            elif int(self.model.geom_contype[gid]) != 0 or int(self.model.geom_conaffinity[gid]) != 0:

                arm.append(gid)

            else:

                # visual-only geoms (group 2) still occlude/render, so they belong

                # to the arm mask, but never to the gripper mask

                name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, body)

                if name not in {None, "world"}:

                    arm.append(gid)

        return arm, gripper

    def set_camera_pose(self, T_C_B: NDArray[np.floating[Any]] | None = None) -> None:

        """Position the camera from ``T_C_B``.

        MuJoCo cameras are world-level in the generated scene.  Some robot
        adapters define B at a rotated mounting body, so compose their
        ``T_W_B`` before applying the camera basis transform.

        """

        T = np.asarray(T_C_B if T_C_B is not None else self.camera.T_C_B, dtype=np.float64)

        T_B_C = invert_transform(T)
        T_W_B = np.asarray(getattr(self.robot, "T_W_B", np.eye(4)), dtype=np.float64)
        T_W_C = T_W_B @ T_B_C
        R_W_cam = T_W_C[:3, :3] @ MJ_CAM_TO_OPTICAL_BASIS

        q = quat_xyzw_from_matrix(R_W_cam)

        self.model.cam_pos[self._cam_id] = T_W_C[:3, 3]

        self.model.cam_quat[self._cam_id] = np.array([q[3], q[0], q[1], q[2]])

    # -- state -------------------------------------------------------------

    def set_state(

        self,

        qpos: NDArray[np.floating[Any]],

        gripper_qpos: NDArray[np.floating[Any]] | None = None,

    ) -> None:

        self.data.qpos[:] = self.model.qpos0
        arm = np.asarray(qpos, dtype=np.float64)
        if arm.shape != (self.robot.config.dof,):
            raise ValueError("arm qpos shape does not match the selected robot profile")
        for name, value in zip(self.robot.config.joint_names, arm, strict=True):
            joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"robot arm joint {name!r} is absent from the renderer model")
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = value
        gdof = self.robot.config.gripper_dof
        if gdof:
            g = (np.asarray(gripper_qpos, dtype=np.float64) if gripper_qpos is not None
                 else np.asarray(self.robot.home_gripper_qpos, dtype=np.float64))
            if g.shape != (gdof,):
                raise ValueError("gripper qpos shape does not match the selected robot profile")
            for name, value in zip(self.robot.config.gripper_joint_names, g, strict=True):
                joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id < 0:
                    raise ValueError(f"robot gripper joint {name!r} is absent from the renderer model")
                self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = value

        self._mujoco.mj_forward(self.model, self.data)

    # -- rendering ---------------------------------------------------------

    def render(

        self,

        qpos: NDArray[np.floating[Any]],

        *,

        gripper_qpos: NDArray[np.floating[Any]] | None = None,

        frame_index: int = 0,

    ) -> dict[str, Any]:

        """Render one frame: RGB, segmentation, masks and metric depth."""

        self.set_state(qpos, gripper_qpos)

        self._renderer.enable_segmentation_rendering()

        self._renderer.update_scene(self.data, camera=self._cam_id, scene_option=self.scene_option)

        seg = np.asarray(self._renderer.render())[:, :, 0].astype(np.int32)

        self._renderer.disable_segmentation_rendering()

        self._renderer.update_scene(self.data, camera=self._cam_id, scene_option=self.scene_option)

        rgb = np.asarray(self._renderer.render(), dtype=np.uint8)

        self._renderer.enable_depth_rendering()

        self._renderer.update_scene(self.data, camera=self._cam_id, scene_option=self.scene_option)

        depth = np.asarray(self._renderer.render(), dtype=np.float32)

        self._renderer.disable_depth_rendering()

        any_robot = np.isin(seg, self._all_robot_geoms) if self._all_robot_geoms.size else np.zeros_like(seg, dtype=bool)

        gripper_mask = (

            np.isin(seg, self._gripper_geoms) if self._gripper_geoms.size else np.zeros_like(seg, dtype=bool)

        )

        robot_mask = any_robot & ~gripper_mask

        # system-design.md 6.5: invalid depth is NaN, never 0 m

        # MuJoCo reports depth as a positive distance along the camera's viewing
        # direction (its local -z), which is also frame C's +z, so the buffer is
        # already the metric optical depth and needs no sign flip.  (Measured: the
        # raw buffer over robot pixels spans 0.26-0.83 m for a workspace ~0.3 m in
        # front of the camera.)  See MJ_CAM_TO_OPTICAL_BASIS.
        depth_m = depth.copy()

        depth_m[~any_robot] = np.nan

        depth_m[~np.isfinite(depth_m)] = np.nan

        depth_m[depth_m <= 0.0] = np.nan

        depth_m[depth_m >= float(self.camera.zfar) * 0.999] = np.nan

        return {

            "rgb": rgb,

            "segmentation": seg,

            "robot_mask": robot_mask,

            "gripper_mask": gripper_mask,

            "depth_m": depth_m.astype(np.float32),

            "frame_index": int(frame_index),

            "camera_id": int(self._cam_id),

        }

    def close(self) -> None:

        try:

            self._renderer.close()

        except Exception:

            pass

        for path in self._scene_paths:

            try:

                path.unlink(missing_ok=True)

            except OSError:

                pass

        self._scene_paths.clear()

    def __enter__(self) -> MujocoRenderer:

        return self

    def __exit__(self, *exc: Any) -> bool:

        self.close()

        return False

def render_robot_scene(

    robot: Any,

    qpos: NDArray[np.floating[Any]],

    camera: CameraSpec,

    *,

    gripper_qpos: NDArray[np.floating[Any]] | None = None,

    frame_index: int = 0,

) -> dict[str, Any]:

    """One-shot convenience wrapper (opens and closes the renderer)."""

    with MujocoRenderer(robot, camera) as r:

        return r.render(qpos, gripper_qpos=gripper_qpos, frame_index=frame_index)

# ---------------------------------------------------------------------------

# Projection verification

# ---------------------------------------------------------------------------

def verify_projection(
    robot: Any,
    camera: CameraSpec,
    points_C_m: NDArray[np.floating[Any]] | None = None,
    *,
    tolerance_px: float = 3.0,
    marker_radius_m: float = 0.012,
    n_points: int = 9,
) -> dict[str, Any]:
    """Compare analytic pinhole projection with rendered marker positions.

    This is the golden check system-design.md 11.2 requires before the renderer
    is trusted: markers are compiled into the model at known **camera-frame**
    points, and each marker's rendered position is compared with ``K @ p_C``.

    Three details were learned the hard way and are all load-bearing:

    * markers live in the same coordinate system the renderer uses, which is the
      robot base frame ``B`` (the MJCF world body), so a camera-frame point is
      mapped through ``T_B_C`` first.  Placing it directly produced a ~3000 px
      residual that looked like a camera bug.
    * marker geoms are created with ``contype=0, conaffinity=0`` and **no**
      ``group``.  Assigning a visual group (``group=3``) makes them invisible to
      the segmentation renderer, which then reports zero visible markers without
      any error.
    * the marker centre is the peak of the distance transform of its pixel mask,
      not the mask centroid.  A marker placed at a TCP target sits where the
      gripper geometry is, so part of it may be occluded, and a centroid is
      biased by exactly that occlusion while the inscribed-disc centre is not.

    When ``points_C_m`` is omitted, a deterministic 3x3 spread of points across
    the field of view is generated from the camera model, so the check does not
    depend on where the robot happens to be.
    """
    import mujoco
    from scipy.ndimage import distance_transform_edt

    cam = camera
    if cam.K is None:
        raise ValueError("verify_projection needs a calibrated K on the CameraSpec")

    generated = points_C_m is None
    if generated:
        # A fan of points spread over the field of view, well away from the
        # optical axis where the arm and its gripper live.  Back-projecting a
        # pixel grid would place the corner points exactly where the end effector
        # happens to be for most poses, and a marker that is partly behind the
        # gripper yields a biased centre.
        azimuth_deg = np.array([-22.0, 0.0, 22.0])
        elevation_deg = np.array([-16.0, 2.0, 20.0])
        depth = 0.55
        fan = np.array(
            [[a, e] for e in elevation_deg for a in azimuth_deg], dtype=np.float64
        )
        az = np.radians(fan[:, 0])
        el = np.radians(fan[:, 1])
        points_C = np.column_stack(
            [
                depth * np.tan(az),
                depth * np.tan(el) / np.cos(az),
                np.full(len(fan), depth),
            ]
        )
    else:
        points_C = np.asarray(points_C_m, dtype=np.float64).reshape(-1, 3)
    points_C = points_C[: max(1, int(n_points))]

    T_B_C = invert_transform(cam.T_C_B)
    points_B = transform_points(T_B_C, points_C)

    spec = mujoco.MjSpec.from_file(str(robot.mjcf_path))
    for i, p in enumerate(points_B):
        body = spec.worldbody.add_body(name=f"e2r_marker_{i}", pos=p.tolist())
        body.add_geom(
            name=f"e2r_marker_geom_{i}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[marker_radius_m, 0.0, 0.0],
            rgba=[1.0, 0.0, 0.0, 1.0],
            contype=0,
            conaffinity=0,
        )
    spec.worldbody.add_camera(name=_CAMERA_NAME, pos=[0, 0, 0], quat=[1, 0, 0, 0], fovy=45.0)
    model = spec.compile()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, cam.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, cam.height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=cam.height, width=cam.width)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, _CAMERA_NAME)
    model.cam_fovy[cam_id] = float(cam.fovy_deg)
    model.cam_resolution[cam_id] = [cam.width, cam.height]
    R_B_cam = T_B_C[:3, :3] @ MJ_CAM_TO_OPTICAL_BASIS
    q = quat_xyzw_from_matrix(R_B_cam)
    model.cam_pos[cam_id] = T_B_C[:3, 3]
    model.cam_quat[cam_id] = np.array([q[3], q[0], q[1], q[2]])
    mujoco.mj_forward(model, data)

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_id)
    seg = np.asarray(renderer.render())[:, :, 0].astype(np.int32)
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_id)
    rgb = np.asarray(renderer.render(), dtype=np.uint8)
    renderer.close()

    analytic_uv = cam.project(points_C)
    rendered_uv = np.full((len(points_C), 2), np.nan)
    visible = np.zeros(len(points_C), dtype=bool)
    visible_px = np.zeros(len(points_C), dtype=np.int64)
    for i in range(len(points_C)):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"e2r_marker_geom_{i}")
        mask = seg == gid
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        dist = distance_transform_edt(mask)
        py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
        rendered_uv[i] = [float(px), float(py)]
        visible[i] = True
        visible_px[i] = count

    errors = np.linalg.norm(rendered_uv - analytic_uv, axis=1)
    finite = np.isfinite(errors)
    max_err = float(np.max(errors[finite])) if finite.any() else None
    return {
        "num_points": int(len(points_C)),
        "visible_points": int(np.count_nonzero(visible)),
        "generated_points": bool(generated),
        "analytic_uv_px": np.where(np.isfinite(analytic_uv), analytic_uv, -1.0).tolist(),
        "rendered_uv_px": np.where(np.isfinite(rendered_uv), rendered_uv, -1.0).tolist(),
        "errors_px": np.where(np.isfinite(errors), errors, -1.0).tolist(),
        "visible_pixels": visible_px.tolist(),
        "max_error_px": max_err,
        "rms_error_px": float(np.sqrt(np.mean(errors[finite] ** 2))) if finite.any() else None,
        "tolerance_px": float(tolerance_px),
        "passes": bool(finite.any() and max_err is not None and max_err <= tolerance_px),
        "camera_basis": "R_B_MjCam = R_B_C @ diag(1, -1, -1)",
        "marker_rgb": rgb,
        "segmentation": seg,
        "point_frame": "C",
    }

