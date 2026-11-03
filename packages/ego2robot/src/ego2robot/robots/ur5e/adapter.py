"""Arm-only Universal Robots UR5e adapter for MuJoCo.

The vendored model is the official Google DeepMind MuJoCo Menagerie asset at
commit ``c96a32d28fb5da84da38c1da4d749e7a13212855``.  It contains six arm
joints and the manufacturer's tool attachment site, but no gripper geometry
or gripper joints.  The FK/IK endpoint exposed here is therefore that official
attachment frame, not a jaw-centre grasp TCP.  Width conversion deliberately
raises instead of inventing a gripper mapping.

Frame contract
--------------
* ``W`` is the MuJoCo world frame.
* ``B`` is the UR5e ``base`` body frame (the model rotates it by 180 degrees
  around Z relative to W).
* ``E`` is ``wrist_3_link``.
* ``TCP`` denotes the model's ``attachment_site`` mounting frame only.  It
  carries no gripper or jaw semantics in this arm-only adapter.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...core.schema import RobotConfig
from ...core.transforms import (
    geodesic_distance_rad,
    invert_transform,
    make_transform,
)
from ..base import CollisionResult, IKResult, RenderFrame, SE3

__all__ = [
    "UR5E_HOME_QPOS",
    "UR5E_JOINT_NAMES",
    "UR5E_MAX_REACH_M",
    "UR5e",
    "UR5eIKSolver",
    "build_ur5e_config",
    "default_ur5e_xml",
]

UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Home is read from the Menagerie MJCF keyframe when building RobotConfig.
# This constant is kept as a reference for users and is checked against it.
UR5E_HOME_QPOS = np.array(
    [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64
)

# Manufacturer technical specification, not a value inferred from a guessed
# DH model.  The MJCF's tool attachment site is used only as the arm endpoint.
UR5E_MAX_REACH_M = 0.850

UR5E_MENAGERIE_REVISION = "c96a32d28fb5da84da38c1da4d749e7a13212855"
UR5E_LICENSE = "BSD-3-Clause"
UR5E_SOURCE = (
    "google-deepmind/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
    f"@{UR5E_MENAGERIE_REVISION}"
)

_ADJACENT_BODY_PAIRS = {
    frozenset({"base", "shoulder_link"}),
    frozenset({"shoulder_link", "upper_arm_link"}),
    frozenset({"upper_arm_link", "forearm_link"}),
    frozenset({"forearm_link", "wrist_1_link"}),
    frozenset({"wrist_1_link", "wrist_2_link"}),
    frozenset({"wrist_2_link", "wrist_3_link"}),
}

_MODEL_DIR = Path(__file__).resolve().parents[3].parent / "assets" / "mjcf" / "universal_robots_ur5e"


def default_ur5e_xml(repo_root: str | Path | None = None) -> Path:
    """Locate the pinned, vendored Menagerie ``ur5e.xml``."""
    if repo_root is not None:
        candidate = Path(repo_root) / "assets" / "mjcf" / "universal_robots_ur5e" / "ur5e.xml"
        if candidate.is_file():
            return candidate
    candidate = _MODEL_DIR / "ur5e.xml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "cannot find assets/mjcf/universal_robots_ur5e/ur5e.xml; "
        "restore the pinned MuJoCo Menagerie model"
    )


def _named_id(mujoco: Any, model: Any, obj: Any, name: str) -> int:
    idx = int(mujoco.mj_name2id(model, obj, name))
    if idx < 0:
        raise ValueError(f"UR5e MJCF is missing required named object {name!r}")
    return idx


def _body_pose_from_data(mujoco: Any, model: Any, data: Any, name: str) -> SE3:
    body_id = _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
    return make_transform(data.xmat[body_id].reshape(3, 3), data.xpos[body_id])


def _site_pose_from_data(mujoco: Any, model: Any, data: Any, name: str) -> SE3:
    site_id = _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, name)
    return make_transform(data.site_xmat[site_id].reshape(3, 3), data.site_xpos[site_id])


def _model_description(mjcf_uri: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    """Compile the source MJCF and derive all model-owned kinematic fields."""
    import mujoco

    path = Path(mjcf_uri)
    if not path.is_file():
        raise FileNotFoundError(f"UR5e MJCF not found: {path}")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)

    joint_names: list[str] = []
    joint_ids: list[int] = []
    qpos_addresses: list[int] = []
    ranges: list[list[float]] = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise ValueError(f"UR5e joint {joint_id} has no name")
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"UR5e joint {name!r} is not a single-DOF hinge joint")
        joint_names.append(name)
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        ranges.append([float(x) for x in model.jnt_range[joint_id]])

    if joint_names != UR5E_JOINT_NAMES:
        raise ValueError(
            "pinned UR5e joint order changed: "
            f"expected {UR5E_JOINT_NAMES}, got {joint_names}"
        )
    if model.nq != len(UR5E_JOINT_NAMES) or model.nv != len(UR5E_JOINT_NAMES):
        raise ValueError(
            f"expected a six-joint arm-only model, got nq={model.nq}, nv={model.nv}"
        )

    home_key_id = _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_KEY, "home")
    home_full = np.asarray(model.key_qpos[home_key_id], dtype=np.float64)
    home_qpos = np.asarray([home_full[adr] for adr in qpos_addresses], dtype=np.float64)
    if not np.allclose(home_qpos, UR5E_HOME_QPOS, atol=1e-5, rtol=0.0):
        raise ValueError(
            "pinned UR5e home keyframe changed: "
            f"expected {UR5E_HOME_QPOS.tolist()}, got {home_qpos.tolist()}"
        )

    base_id = _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "base")
    eef_id = _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    site_id = _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    mujoco.mj_forward(model, data)
    T_W_B = make_transform(data.xmat[base_id].reshape(3, 3), data.xpos[base_id])
    T_W_E = make_transform(data.xmat[eef_id].reshape(3, 3), data.xpos[eef_id])
    T_W_attachment = make_transform(
        data.site_xmat[site_id].reshape(3, 3), data.site_xpos[site_id]
    )
    T_E_attachment = invert_transform(T_W_E) @ T_W_attachment

    actuator_names: list[str] = []
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name is None:
            raise ValueError(f"UR5e actuator {actuator_id} has no name")
        actuator_names.append(name)

    description = {
        "joint_names": joint_names,
        "joint_ids": joint_ids,
        "qpos_addresses": qpos_addresses,
        "position_limits": ranges,
        "home_qpos": home_qpos,
        "T_W_B": T_W_B,
        "T_E_attachment": T_E_attachment,
        "actuator_names": actuator_names,
        "has_gripper": False,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nsite": int(model.nsite),
        "nu": int(model.nu),
    }
    return mujoco, model, description


def build_ur5e_config(
    *,
    mjcf_uri: str | Path,
    robot_id: str = "ur5e",
    profile: Any | None = None,
) -> RobotConfig:
    """Construct a RobotConfig from the pinned arm-only Menagerie model.

    ``T_E_TCP`` is derived from the official ``attachment_site`` in the MJCF.
    This adapter has no jaw frame, gripper joints, gripper geometry, or width
    mapping.  Profile claims which imply a gripper are rejected.
    """
    path = Path(mjcf_uri)
    mujoco, model, description = _model_description(path)
    del mujoco
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    derived: dict[str, Any] = {
        "dof": 6,
        "gripper_dof": 0,
        "max_reach_m": UR5E_MAX_REACH_M,
        "eef_body_name": "wrist_3_link",
        "base_body_name": "base",
        "joint_names": description["joint_names"],
        "gripper_joint_names": [],
    }
    if profile is not None:
        mismatches: list[str] = []
        for field, expected in derived.items():
            declared = getattr(profile, field, None)
            if declared is None or declared == [] or declared == "":
                continue
            if declared != expected:
                mismatches.append(
                    f"{field}: profile says {declared!r}, model adapter declares {expected!r}"
                )
        declared_source = getattr(profile, "mjcf_source", "")
        if declared_source and declared_source != UR5E_SOURCE:
            mismatches.append(
                f"mjcf_source: profile says {declared_source!r}, pinned source is {UR5E_SOURCE!r}"
            )
        declared_license = getattr(profile, "mjcf_license", "")
        if declared_license and declared_license != UR5E_LICENSE:
            mismatches.append(
                f"mjcf_license: profile says {declared_license!r}, pinned source is {UR5E_LICENSE!r}"
            )
        if mismatches:
            raise ValueError("UR5e robot profile mismatch: " + "; ".join(mismatches))
        width_max = getattr(profile, "gripper_width_max_m", 0.0)
        width_min = getattr(profile, "gripper_width_min_m", 0.0)
        if width_max not in (None, 0, 0.0) or width_min not in (None, 0, 0.0):
            raise ValueError(
                "UR5e profile requests jaw width limits, but the official arm-only "
                "MJCF has no gripper; set both gripper_width_min_m and "
                "gripper_width_max_m to 0"
            )

    # The official model permits elbow_joint in [-3.1415, +3.1415] rad, while
    # the manufacturer's technical sheet specifies +/-160 degrees.  Keep the
    # vendored model's own range here; choosing a hardware-safe override is an
    # open project decision and must be recorded in an ADR.
    notes = [
        "Arm-only model: official MJCF contains six revolute arm joints and no gripper bodies, joints, or actuators.",
        "TCP is the MJCF attachment_site mounting frame on wrist_3_link; it is not a jaw-centre grasp frame.",
        "No gripper width mapping or gripper geometry is available; gripper_dof=0.",
        "max_reach_m=0.850 from the Universal Robots UR5e technical specification (850 mm).",
        "position_limits are copied from the pinned MJCF; its elbow range differs from the manufacturer's +/-160-degree specification and needs an ADR before being treated as a hardware limit.",
        f"T_E_TCP derived from the official attachment_site in Menagerie revision {UR5E_MENAGERIE_REVISION}.",
        f"Vendored MJCF SHA-256: {sha256}.",
    ]
    config = RobotConfig(
        robot_id=robot_id or "ur5e",
        dof=derived["dof"],
        gripper_dof=0,
        max_reach_m=derived["max_reach_m"],
        home_qpos=description["home_qpos"].tolist(),
        home_gripper_qpos=[],
        mjcf_uri=str(path),
        mjcf_sha256=sha256,
        mjcf_license=UR5E_LICENSE,
        mjcf_source=UR5E_SOURCE,
        T_E_TCP=description["T_E_attachment"].tolist(),
        joint_names=description["joint_names"],
        gripper_joint_names=[],
        actuator_names=description["actuator_names"],
        position_limits=description["position_limits"],
        eef_body_name=derived["eef_body_name"],
        base_body_name=derived["base_body_name"],
        gripper_geom_names=[],
        notes=notes,
    )
    return config


@dataclass
class UR5eIKSolver:
    """Mink differential IK targeting the wrist body via attachment_site."""

    model: Any
    config: RobotConfig
    T_W_B: SE3
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    lm_damping: float = 0.0
    iterations: int = 100
    tolerance: float = 1e-5
    solver: str = "quadprog"

    def __post_init__(self) -> None:
        import mink
        import mujoco

        self._mink = mink
        self._mujoco = mujoco
        self.task = mink.FrameTask(
            frame_name=self.config.eef_body_name,
            frame_type="body",
            position_cost=float(self.position_cost),
            orientation_cost=float(self.orientation_cost),
            lm_damping=float(self.lm_damping),
        )
        self.limits = [mink.ConfigurationLimit(self.model)]
        self._configuration = mink.Configuration(self.model)
        self._q_arm = np.asarray(self.config.home_qpos, dtype=np.float64)

    def solve(
        self,
        target_T_B_TCP: SE3,
        q_init: NDArray[np.floating[Any]] | None = None,
        *,
        max_iterations: int | None = None,
    ) -> IKResult:
        target_B_TCP = np.asarray(target_T_B_TCP, dtype=np.float64).reshape(4, 4)
        target_W_TCP = self.T_W_B @ target_B_TCP
        target_W_E = target_W_TCP @ invert_transform(self.config.T_E_TCP_array)
        seed = (
            np.asarray(q_init, dtype=np.float64).reshape(-1)
            if q_init is not None
            else self._q_arm.copy()
        )
        if seed.shape != (self.config.dof,):
            raise ValueError(f"UR5e q_init must have shape ({self.config.dof},), got {seed.shape}")
        config = self._configuration
        config.update(seed.copy())
        self.task.set_target(self._mink.SE3.from_matrix(target_W_E))

        iterations = int(max_iterations if max_iterations is not None else self.iterations)
        status = "max_iterations"
        used = 0
        error_msg: str | None = None
        for step in range(1, iterations + 1):
            used = step
            try:
                velocity = self._mink.solve_ik(
                    config,
                    [self.task],
                    dt=1e-2,
                    solver=self.solver,
                    damping=1e-5,
                    limits=self.limits,
                )
            except self._mink.NoSolutionFound:
                status = "no_solution"
                break
            except Exception as exc:  # noqa: BLE001 - returned as an IK status
                status = "error"
                error_msg = f"{type(exc).__name__}: {exc}"
                break
            config.integrate_inplace(velocity, dt=1e-2)
            residual = self.task.compute_error(config)
            if float(np.linalg.norm(residual[:3])) < self.tolerance and float(
                np.linalg.norm(residual[3:])
            ) < self.tolerance:
                status = "converged"
                break

        q_solution = np.asarray(config.q[: self.config.dof], dtype=np.float64).copy()
        T_W_E_fk = _body_pose_from_data(
            self._mujoco, self.model, self._data_for(q_solution), self.config.eef_body_name
        )
        T_B_TCP_fk = invert_transform(self.T_W_B) @ T_W_E_fk @ self.config.T_E_TCP_array
        pos_error = float(np.linalg.norm(T_B_TCP_fk[:3, 3] - target_B_TCP[:3, 3]))
        rot_error = geodesic_distance_rad(T_B_TCP_fk[:3, :3], target_B_TCP[:3, :3])
        limit_hit = self._limits_hit(q_solution)
        success = bool(pos_error <= 1e-4 and rot_error <= 1e-3 and status == "converged")
        return IKResult(
            success=success,
            qpos=q_solution,
            T_B_TCP_fk=T_B_TCP_fk,
            position_error_m=pos_error,
            rotation_error_rad=rot_error,
            iterations=used,
            status=status,
            status_code={"converged": 0, "max_iterations": 1, "no_solution": 2, "error": 3}.get(status, 9),
            joint_limit_hit=limit_hit,
            error=error_msg,
        )

    def _data_for(self, qpos: NDArray[np.float64]) -> Any:
        import mujoco

        data = mujoco.MjData(self.model)
        data.qpos[:] = 0.0
        data.qpos[: self.config.dof] = qpos
        mujoco.mj_forward(self.model, data)
        return data

    def _limits_hit(self, qpos: NDArray[np.float64]) -> bool:
        import mujoco

        for index, name in enumerate(self.config.joint_names):
            jid = _named_id(self._mujoco, self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = self.model.jnt_range[jid]
            if qpos[index] < lo - 1e-6 or qpos[index] > hi + 1e-6:
                return True
        return False


class UR5e:
    """Six-axis UR5e arm using the Menagerie ``attachment_site`` endpoint."""

    robot_id = "ur5e"
    has_gripper = False
    supports_gripper = False
    endpoint_semantics = "tool_mount_only_no_gripper"

    def __init__(
        self,
        mjcf_uri: str | Path | None = None,
        *,
        config: RobotConfig | None = None,
        ik: UR5eIKSolver | None = None,
        ik_settings: dict[str, Any] | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        path = Path(mjcf_uri) if mjcf_uri is not None else default_ur5e_xml()
        if not path.is_file():
            raise FileNotFoundError(f"UR5e MJCF not found: {path}")
        self.mjcf_path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.config = config or build_ur5e_config(mjcf_uri=path)
        self.dof = self.config.dof
        self.max_reach_m = self.config.max_reach_m
        self.home_qpos = np.asarray(self.config.home_qpos, dtype=np.float64)
        self.home_gripper_qpos = np.empty((0,), dtype=np.float64)
        self.max_gripper_width_m = 0.0

        self._set_qpos(self.home_qpos)
        self.T_W_B = _body_pose_from_data(mujoco, self.model, self.data, self.config.base_body_name)
        self.T_B_W = invert_transform(self.T_W_B)
        settings = dict(ik_settings or {})
        self.ik_solver = ik or UR5eIKSolver(
            model=self.model,
            config=self.config,
            T_W_B=self.T_W_B,
            position_cost=float(settings.get("position_cost", 1.0)),
            orientation_cost=float(settings.get("orientation_cost", 1.0)),
            lm_damping=float(settings.get("lm_damping", 0.0)),
            iterations=int(settings.get("iterations", 100)),
            tolerance=float(settings.get("tolerance", 1e-5)),
            solver=str(settings.get("qp_backend", "quadprog")),
        )

    def _set_qpos(self, qpos: NDArray[np.floating[Any]]) -> None:
        q = np.array(qpos, dtype=np.float64, copy=True).reshape(-1)
        if q.shape != (self.dof,):
            raise ValueError(f"UR5e qpos must have shape ({self.dof},), got {q.shape}")
        self.data.qpos[:] = 0.0
        for i, name in enumerate(self.config.joint_names):
            jid = _named_id(self._mujoco, self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            adr = int(self.model.jnt_qposadr[jid])
            self.data.qpos[adr] = q[i]
        self._mujoco.mj_forward(self.model, self.data)

    def _world_body_pose(self, body_name: str) -> SE3:
        return _body_pose_from_data(self._mujoco, self.model, self.data, body_name)

    def gripper_width_to_qpos(self, width_m: float) -> NDArray[np.float64]:
        """Return the empty actuator vector required by the shared arm stage.

        This does not execute or preserve ``width_m``.  Callers must inspect
        ``has_gripper=False`` / ``supports_gripper=False`` and must not report
        grasp width as realized on the UR5e.
        """
        del width_m
        return np.empty((0,), dtype=np.float64)

    def qpos_to_gripper_width(self, gripper_qpos: NDArray[np.floating[Any]]) -> float:
        del gripper_qpos
        raise NotImplementedError("UR5e Menagerie asset has no gripper joints or jaw width")

    def forward_kinematics(
        self,
        qpos: NDArray[np.floating[Any]],
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> SE3:
        if gripper_qpos is not None and np.asarray(gripper_qpos).size:
            raise ValueError("UR5e is arm-only; gripper_qpos must be empty")
        self._set_qpos(qpos)
        T_W_attachment = _site_pose_from_data(
            self._mujoco, self.model, self.data, "attachment_site"
        )
        return self.T_B_W @ T_W_attachment

    def solve_ik(
        self,
        target_T_B_TCP: SE3,
        q_init: NDArray[np.floating[Any]] | None = None,
        *,
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> IKResult:
        if gripper_qpos is not None and np.asarray(gripper_qpos).size:
            raise ValueError("UR5e is arm-only; gripper_qpos must be empty")
        return self.ik_solver.solve(target_T_B_TCP, q_init)

    def check_collision(
        self,
        qpos: NDArray[np.floating[Any]],
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> CollisionResult:
        if gripper_qpos is not None and np.asarray(gripper_qpos).size:
            raise ValueError("UR5e is arm-only; gripper_qpos must be empty")
        self._set_qpos(qpos)
        result = CollisionResult(has_self_collision=False)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = self._mujoco.mj_id2name(
                self.model,
                self._mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[contact.geom1]),
            ) or "world"
            b2 = self._mujoco.mj_id2name(
                self.model,
                self._mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[contact.geom2]),
            ) or "world"
            pair = (b1, b2)
            result.max_penetration_m = max(result.max_penetration_m, -float(contact.dist))
            if "world" in (b1, b2):
                result.scene_contact_pairs.append(pair)
            elif frozenset({b1, b2}) in _ADJACENT_BODY_PAIRS:
                result.adjacent_contact_pairs.append(pair)
            else:
                result.self_contact_pairs.append(pair)
        result.has_self_collision = bool(result.self_contact_pairs)
        if result.has_self_collision:
            result.code = 1
        elif result.scene_contact_pairs:
            result.code = 2
        return result

    def render(self, qpos: NDArray[np.floating[Any]], camera: Any) -> RenderFrame:
        """Render arm RGB/mask/depth; gripper_mask is explicitly empty."""
        from dataclasses import replace

        from ...vision.render import MujocoRenderer

        T_C_W = np.asarray(camera.T_C_B, dtype=np.float64) @ self.T_B_W
        world_camera = replace(camera, T_C_B=T_C_W)
        renderer = MujocoRenderer(self, world_camera)
        try:
            frame = renderer.render(qpos)
        finally:
            renderer.close()
        return RenderFrame(
            rgb=frame["rgb"],
            robot_mask=frame["robot_mask"],
            gripper_mask=frame["gripper_mask"],
            depth_m=frame["depth_m"],
            frame_index=frame["frame_index"],
            metadata={
                "robot_id": self.robot_id,
                "endpoint_semantics": self.endpoint_semantics,
                "gripper_present": False,
                "render_camera_frame": "MuJoCo world W",
            },
        )

    def body_positions(
        self,
        qpos: NDArray[np.floating[Any]],
        names: list[str],
    ) -> dict[str, NDArray[np.float64]]:
        self._set_qpos(qpos)
        return {name: self._world_body_pose(name)[:3, 3] for name in names}

    def reach_ratio(self, T_B_TCP: SE3) -> float:
        return float(np.linalg.norm(np.asarray(T_B_TCP)[:3, 3])) / self.max_reach_m

    def apply_home(self) -> None:
        self._set_qpos(self.home_qpos)

    def set_state(
        self,
        qpos: NDArray[np.floating[Any]],
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        if gripper_qpos is not None and np.asarray(gripper_qpos).size:
            raise ValueError("UR5e is arm-only; gripper_qpos must be empty")
        self._set_qpos(qpos)

    def body_pose(self, body_name: str) -> SE3:
        return self._world_body_pose(body_name)

    def tcp_pose(self) -> SE3:
        return self.forward_kinematics(self.data.qpos[: self.dof])

    def kinematics_summary(self) -> dict[str, Any]:
        joints: dict[str, Any] = {}
        for i, name in enumerate(self.config.joint_names):
            jid = _named_id(self._mujoco, self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            joints[name] = {
                "range_rad": [float(x) for x in self.model.jnt_range[jid]],
                "qpos_adr": int(self.model.jnt_qposadr[jid]),
                "home_qpos_rad": float(self.home_qpos[i]),
            }
        return {
            "robot_id": self.robot_id,
            "mjcf": str(self.mjcf_path),
            "mjcf_sha256": self.config.mjcf_sha256,
            "mjcf_source": self.config.mjcf_source,
            "license": self.config.mjcf_license,
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "dof": self.dof,
            "gripper_dof": 0,
            "has_gripper": False,
            "max_reach_m": self.max_reach_m,
            "base_body": self.config.base_body_name,
            "eef_body": self.config.eef_body_name,
            "endpoint_site": "attachment_site",
            "endpoint_semantics": self.endpoint_semantics,
            "T_W_B": self.T_W_B.tolist(),
            "T_E_TCP": self.config.T_E_TCP,
            "joints": joints,
            "bodies": [
                self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, i)
                for i in range(self.model.nbody)
            ],
        }
