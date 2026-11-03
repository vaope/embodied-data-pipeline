"""UFACTORY xArm7 + parallel gripper adapter for MuJoCo Menagerie.

The vendored source is pinned to ``google-deepmind/mujoco_menagerie`` commit
``c96a32d28fb5da84da38c1da4d749e7a13212855``.  The project base frame ``B`` is
the included MJCF world's origin (the robot mounting plane); the source model
places ``link_base`` at ``z=0.12 m`` above it.  The end-effector body ``E`` is
``link7`` and TCP is the source model's ``link_tcp`` site.  Its fixed transform
is obtained from the compiled model's FK rather than entered by hand.

The xArm gripper has six generalized hinge coordinates but one tendon actuator.
All six coordinates are coupled by the source model's equality constraints.  A
deterministic calibration table maps canonical jaw width to a common hinge
position, measured between the inner faces of the four source pad boxes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ...core.schema import RobotConfig
from ...core.transforms import invert_transform, make_transform
from ..base import CollisionResult, IKResult, RenderFrame, SE3

__all__ = [
    "XARM7_ARM_JOINT_NAMES",
    "XARM7_GRIPPER_JOINT_NAMES",
    "XARM7_MENAGERIE_COMMIT",
    "XArm7IKSolver",
    "XArm7",
    "build_xarm7_config",
    "default_xarm7_xml",
]

XARM7_MENAGERIE_COMMIT = "c96a32d28fb5da84da38c1da4d749e7a13212855"
XARM7_ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
XARM7_GRIPPER_JOINT_NAMES = (
    "left_driver_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_driver_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)
XARM7_ACTUATOR_NAMES = tuple(f"act{i}" for i in range(1, 8)) + ("gripper",)

_EEF_BODY_NAME = "link7"
_BASE_BODY_NAME = "world"
_GRIPPER_BASE_BODY_NAME = "xarm_gripper_base_link"
_TCP_SITE_NAME = "link_tcp"
_PAD_GEOMS = {
    "left": ("left_finger_pad_1", "left_finger_pad_2"),
    "right": ("right_finger_pad_1", "right_finger_pad_2"),
}
_CALIBRATION_SAMPLES = 257

# Contacts between these parent-child bodies are part of the model's physical
# assembly. The Menagerie XML separately excludes four inner/outer-knuckle and
# inner-knuckle/finger pairs from collision generation.
_ADJACENT_PAIRS = {
    frozenset({"link_base", "link1"}),
    frozenset({"link1", "link2"}),
    frozenset({"link2", "link3"}),
    frozenset({"link3", "link4"}),
    frozenset({"link4", "link5"}),
    frozenset({"link5", "link6"}),
    frozenset({"link6", "link7"}),
    frozenset({"link7", _GRIPPER_BASE_BODY_NAME}),
    frozenset({_GRIPPER_BASE_BODY_NAME, "left_outer_knuckle"}),
    frozenset({_GRIPPER_BASE_BODY_NAME, "left_inner_knuckle"}),
    frozenset({_GRIPPER_BASE_BODY_NAME, "right_outer_knuckle"}),
    frozenset({_GRIPPER_BASE_BODY_NAME, "right_inner_knuckle"}),
    frozenset({"left_outer_knuckle", "left_finger"}),
    frozenset({"left_inner_knuckle", "left_finger"}),
    frozenset({"right_outer_knuckle", "right_finger"}),
    frozenset({"right_inner_knuckle", "right_finger"}),
    frozenset({"left_finger", "right_finger"}),
}

_MODEL_DIR = Path(__file__).resolve().parents[3].parent / "assets" / "mjcf" / "ufactory_xarm7"


def default_xarm7_xml(repo_root: str | Path | None = None) -> Path:
    """Locate the pinned, vendored xArm7 MJCF."""
    if repo_root is not None:
        candidate = Path(repo_root) / "assets" / "mjcf" / "ufactory_xarm7" / "xarm7.xml"
        if candidate.is_file():
            return candidate
    candidate = _MODEL_DIR / "xarm7.xml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "cannot find assets/mjcf/ufactory_xarm7/xarm7.xml; fetch the pinned "
        "MuJoCo Menagerie directory into assets/mjcf/ufactory_xarm7/"
    )


def _name_id(mujoco: Any, model: Any, obj_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, obj_type, name))
    if object_id < 0:
        raise ValueError(f"xArm7 MJCF is missing required {obj_type.name} {name!r}")
    return object_id


def _joint_ids(mujoco: Any, model: Any, names: Sequence[str]) -> list[int]:
    ids = [_name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    if any(int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE) for jid in ids):
        raise ValueError("xArm7 adapter requires every arm and gripper coordinate to be a hinge joint")
    return ids


def _qpos_addresses(model: Any, joint_ids: Sequence[int]) -> NDArray[np.int64]:
    return np.asarray([int(model.jnt_qposadr[jid]) for jid in joint_ids], dtype=np.int64)


def _home_from_keyframe(mujoco: Any, model: Any) -> NDArray[np.float64]:
    key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    if key_id < 0:
        raise ValueError("xArm7 MJCF does not declare the required 'home' keyframe")
    return np.asarray(model.key_qpos[key_id], dtype=np.float64).copy()


def _body_pose_from_data(mujoco: Any, model: Any, data: Any, name: str) -> NDArray[np.float64]:
    body_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
    return make_transform(data.xmat[body_id].reshape(3, 3), data.xpos[body_id])


def _canonicalise_tcp_axes(T_E_TCP: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bring the model's ``link_tcp`` site axes onto the canonical conventions.

    The site defines its own axes. Measured on this model, the site's ``x/y/z``
    correspond to ``link7`` axes ``-x/-y/+z``, i.e. the site's ``z`` points at the
    gripper (approach) and its ``y`` lies along the jaw line. The canonical frame
    instead requires ``x = approach``, ``y = jaw-plane normal``, ``z = grasp axis``
    (system-design.md 6.4, paper equation 3).

    Composing with the permutation below fixes all three at once, keeping
    ``det = +1``. Verified with MuJoCo FK: driving both linkage drivers moves the
    two finger bodies along the resulting ``z`` axis (0.0°), whereas before the fix
    they moved along ``y`` (90° off). See ADR-0005 (amendment, 2026-10-27).
    """
    fix = np.array(
        [[0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    out = np.array(T_E_TCP, dtype=np.float64, copy=True)
    out[:3, :3] = out[:3, :3] @ fix
    return out


def _tcp_transform_in_eef(
    mujoco: Any,
    model: Any,
    home_qpos: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Derive ``T_E_TCP`` from the compiled model's link_tcp site and FK."""
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    mujoco.mj_forward(model, data)
    eef = _body_pose_from_data(mujoco, model, data, _EEF_BODY_NAME)
    site_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, _TCP_SITE_NAME)
    tcp = make_transform(data.site_xmat[site_id].reshape(3, 3), data.site_xpos[site_id])
    return invert_transform(eef) @ tcp


def _pad_gap_at_angle(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    arm_qpos: NDArray[np.float64],
    arm_qpos_addresses: NDArray[np.int64],
    gripper_qpos_addresses: NDArray[np.int64],
    angle: float,
) -> float:
    """Measure jaw opening from inner pad-box faces in the gripper base frame."""
    data.qpos[:] = model.qpos0
    data.qpos[arm_qpos_addresses] = arm_qpos
    data.qpos[gripper_qpos_addresses] = float(angle)
    mujoco.mj_forward(model, data)

    base_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, _GRIPPER_BASE_BODY_NAME)
    base_pos = np.asarray(data.xpos[base_id], dtype=np.float64)
    base_R_world = np.asarray(data.xmat[base_id], dtype=np.float64).reshape(3, 3)
    inward_faces: dict[str, list[float]] = {"left": [], "right": []}
    signs = np.asarray(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )

    for side, names in _PAD_GEOMS.items():
        for geom_name in names:
            geom_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            center_world = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            R_world_geom = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            half_sizes = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            corners_world = center_world + (signs * half_sizes) @ R_world_geom.T
            corners_base = (corners_world - base_pos) @ base_R_world
            face = float(np.min(corners_base[:, 1])) if side == "left" else float(np.max(corners_base[:, 1]))
            inward_faces[side].append(face)
    return float(np.mean(inward_faces["left"]) - np.mean(inward_faces["right"]))


def _gripper_calibration(
    mujoco: Any,
    model: Any,
    home_qpos: NDArray[np.float64],
    arm_joint_ids: Sequence[int],
    gripper_joint_ids: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    arm_addresses = _qpos_addresses(model, arm_joint_ids)
    gripper_addresses = _qpos_addresses(model, gripper_joint_ids)
    ranges = np.asarray([model.jnt_range[jid] for jid in gripper_joint_ids], dtype=np.float64)
    lower = float(np.max(ranges[:, 0]))
    upper = float(np.min(ranges[:, 1]))
    if not lower < upper:
        raise ValueError("xArm7 gripper joint ranges have no common position interval")

    # The upstream equality constraints tie all six gripper coordinates to one
    # common angle. Verify that invariant before using one scalar calibration.
    key_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_KEY, "home")
    key_qpos = np.asarray(model.key_qpos[key_id], dtype=np.float64)
    if not np.allclose(key_qpos[gripper_addresses], key_qpos[gripper_addresses[0]], atol=1e-9):
        raise ValueError("xArm7 home keyframe violates the common-angle gripper coordinate contract")

    angles = np.linspace(lower, upper, _CALIBRATION_SAMPLES, dtype=np.float64)
    widths = np.empty_like(angles)
    data = mujoco.MjData(model)
    widths[:] = [
        _pad_gap_at_angle(
            mujoco,
            model,
            data,
            arm_qpos=home_qpos[arm_addresses],
            arm_qpos_addresses=arm_addresses,
            gripper_qpos_addresses=gripper_addresses,
            angle=float(angle),
        )
        for angle in angles
    ]
    if np.any(widths <= 0.0):
        raise ValueError("xArm7 gripper pad geometry overlaps within its declared joint range")
    if not np.all(np.diff(widths) < 0.0):
        raise ValueError("xArm7 gripper pad opening is not strictly monotone across joint limits")
    return angles, widths


@lru_cache(maxsize=8)
def _max_reach_for_model(path_string: str, model_sha256: str) -> float:
    """Numerically maximize TCP distance over parsed arm-joint limits.

    The optimizer settings and seeds are fixed for reproducibility. The model
    hash is part of the cache key, so updating the vendored asset invalidates the
    cached result.
    """
    import mujoco
    from scipy.optimize import differential_evolution

    path = Path(path_string)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    joint_ids = _joint_ids(mujoco, model, XARM7_ARM_JOINT_NAMES)
    addresses = _qpos_addresses(model, joint_ids)
    bounds = [tuple(float(v) for v in model.jnt_range[jid]) for jid in joint_ids]
    site_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, _TCP_SITE_NAME)

    def objective(q_arm: NDArray[np.float64]) -> float:
        data.qpos[:] = model.qpos0
        data.qpos[addresses] = q_arm
        mujoco.mj_forward(model, data)
        return -float(np.linalg.norm(data.site_xpos[site_id]))

    distances: list[float] = []
    for seed in (7, 17, 23):
        result = differential_evolution(
            objective,
            bounds,
            seed=seed,
            popsize=16,
            maxiter=100,
            tol=1e-8,
            polish=True,
            workers=1,
            updating="immediate",
        )
        if not result.success:
            raise RuntimeError(
                f"xArm7 max-reach derivation did not converge for deterministic seed {seed}: "
                f"{result.message}"
            )
        distances.append(-float(result.fun))
    return float(max(distances))


def _model_contract(mjcf_uri: str | Path) -> dict[str, Any]:
    import mujoco

    path = Path(mjcf_uri)
    if not path.is_file():
        raise FileNotFoundError(f"xArm7 MJCF not found: {path}")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    model = mujoco.MjModel.from_xml_path(str(path))
    arm_ids = _joint_ids(mujoco, model, XARM7_ARM_JOINT_NAMES)
    gripper_ids = _joint_ids(mujoco, model, XARM7_GRIPPER_JOINT_NAMES)
    arm_addresses = _qpos_addresses(model, arm_ids)
    gripper_addresses = _qpos_addresses(model, gripper_ids)
    if model.nq != len(arm_ids) + len(gripper_ids):
        raise ValueError(
            f"expected exactly {len(arm_ids) + len(gripper_ids)} hinge qpos values, got nq={model.nq}"
        )
    if model.nu != len(XARM7_ACTUATOR_NAMES):
        raise ValueError(f"expected {len(XARM7_ACTUATOR_NAMES)} actuators, got nu={model.nu}")
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(model.nu)
    )
    if actuator_names != XARM7_ACTUATOR_NAMES:
        raise ValueError(f"unexpected xArm7 actuator order/names: {actuator_names!r}")
    gripper_actuator_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    if int(model.actuator_trntype[gripper_actuator_id]) != int(mujoco.mjtTrn.mjTRN_TENDON):
        raise ValueError("xArm7 gripper actuator is no longer a tendon actuator")

    home_all = _home_from_keyframe(mujoco, model)
    angles, widths = _gripper_calibration(
        mujoco, model, home_all, arm_ids, gripper_ids
    )
    limits = np.asarray([model.jnt_range[jid] for jid in arm_ids], dtype=np.float64)
    return {
        "path": path,
        "sha256": sha,
        "model": model,
        "arm_ids": arm_ids,
        "gripper_ids": gripper_ids,
        "arm_addresses": arm_addresses,
        "gripper_addresses": gripper_addresses,
        "home_all": home_all,
        "angles": angles,
        "widths": widths,
        "limits": limits,
        "actuator_names": list(actuator_names),
        "max_reach_m": _max_reach_for_model(str(path.resolve()), sha),
    }


def build_xarm7_config(
    *,
    mjcf_uri: str | Path,
    robot_id: str = "xarm7",
    profile: Any | None = None,
) -> RobotConfig:
    """Build and validate a persisted ``RobotConfig`` from the pinned MJCF."""
    contract = _model_contract(mjcf_uri)
    limits = contract["limits"]
    derived: dict[str, Any] = {
        "dof": len(XARM7_ARM_JOINT_NAMES),
        "gripper_dof": len(XARM7_GRIPPER_JOINT_NAMES),
        "max_reach_m": contract["max_reach_m"],
        "eef_body_name": _EEF_BODY_NAME,
        "base_body_name": _BASE_BODY_NAME,
        "joint_names": list(XARM7_ARM_JOINT_NAMES),
        "gripper_joint_names": list(XARM7_GRIPPER_JOINT_NAMES),
        "gripper_width_min_m": float(contract["widths"][-1]),
        "gripper_width_max_m": float(contract["widths"][0]),
    }
    if profile is not None:
        mismatches: list[str] = []
        for field, value in derived.items():
            declared = getattr(profile, field, None)
            if declared is None or declared == [] or declared == "":
                continue
            if field.endswith("_m") and isinstance(declared, (int, float)):
                if not np.isclose(float(declared), float(value), rtol=0.0, atol=1e-3):
                    mismatches.append(f"{field}: profile says {declared!r}, model derives {value!r}")
            elif declared != value:
                mismatches.append(f"{field}: profile says {declared!r}, model derives {value!r}")
        if mismatches:
            raise ValueError("robot profile disagrees with the vendored xArm7 MJCF: " + "; ".join(mismatches))

    arm_home = contract["home_all"][contract["arm_addresses"]]
    gripper_home = contract["home_all"][contract["gripper_addresses"]]
    T_E_TCP = _tcp_transform_in_eef(
        __import__("mujoco"), contract["model"], contract["home_all"]
    )
    T_E_TCP = _canonicalise_tcp_axes(T_E_TCP)
    return RobotConfig(
        robot_id=robot_id or (getattr(profile, "robot_id", None) or "xarm7"),
        dof=derived["dof"],
        gripper_dof=derived["gripper_dof"],
        max_reach_m=float(derived["max_reach_m"]),
        home_qpos=arm_home.tolist(),
        home_gripper_qpos=gripper_home.tolist(),
        mjcf_uri=str(contract["path"]),
        mjcf_sha256=contract["sha256"],
        mjcf_license=str(getattr(profile, "mjcf_license", "") or "BSD-3-Clause"),
        mjcf_source=_source_with_commit(
            str(getattr(profile, "mjcf_source", "") or "mujoco_menagerie/ufactory_xarm7/xarm7.xml")
        ),
        T_E_TCP=T_E_TCP.tolist(),
        joint_names=list(XARM7_ARM_JOINT_NAMES),
        gripper_joint_names=list(XARM7_GRIPPER_JOINT_NAMES),
        actuator_names=list(contract["actuator_names"]),
        position_limits=limits.tolist(),
        eef_body_name=_EEF_BODY_NAME,
        base_body_name=_BASE_BODY_NAME,
        gripper_geom_names=[
            _GRIPPER_BASE_BODY_NAME,
            "left_outer_knuckle",
            "left_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_finger",
            "right_inner_knuckle",
        ],
        max_joint_jump_rad=1.5,
        notes=[
            "B is the MJCF world origin/mounting plane; link_base is at +0.12 m z in the official XML",
            "T_E_TCP is derived from the link_tcp site relative to link7 by MuJoCo FK",
            "gripper_dof=6 generalized hinge coordinates; one separate tendon actuator named gripper",
            "gripper width is calibrated from the four pad-box inner faces over their shared MJCF joint range",
            "max_reach_m is the deterministic differential-evolution maximum TCP radius over parsed arm limits",
        ],
    )


def _source_with_commit(source: str) -> str:
    suffix = f"@{XARM7_MENAGERIE_COMMIT}"
    return source if source.endswith(suffix) else source + suffix


@dataclass
class XArm7IKSolver:
    """Mink IK solver controlling xArm7's ``link7`` body."""

    model: Any
    config: RobotConfig
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
        self._arm_joint_ids = _joint_ids(mujoco, self.model, self.config.joint_names)
        self._gripper_joint_ids = _joint_ids(mujoco, self.model, self.config.gripper_joint_names)
        self._arm_addresses = _qpos_addresses(self.model, self._arm_joint_ids)
        self._gripper_addresses = _qpos_addresses(self.model, self._gripper_joint_ids)
        self.task = mink.FrameTask(
            frame_name=self.config.eef_body_name,
            frame_type="body",
            position_cost=float(self.position_cost),
            orientation_cost=float(self.orientation_cost),
            lm_damping=float(self.lm_damping),
        )
        self.limits = [mink.ConfigurationLimit(self.model)]
        self._configuration = mink.Configuration(self.model)

    def _set_configuration(
        self,
        q_arm: NDArray[np.float64],
        gripper_qpos: NDArray[np.float64],
    ) -> Any:
        qpos = self._configuration.q.copy()
        qpos[self._arm_addresses] = q_arm
        qpos[self._gripper_addresses] = gripper_qpos
        self._configuration.update(qpos)
        return self._configuration

    def _body_pose(
        self,
        q_arm: NDArray[np.float64],
        gripper_qpos: NDArray[np.float64],
        body_name: str,
    ) -> NDArray[np.float64]:
        data = self._mujoco.MjData(self.model)
        data.qpos[:] = self.model.qpos0
        data.qpos[self._arm_addresses] = q_arm
        data.qpos[self._gripper_addresses] = gripper_qpos
        self._mujoco.mj_forward(self.model, data)
        return _body_pose_from_data(self._mujoco, self.model, data, body_name)

    def _limits_hit(self, q_arm: NDArray[np.float64]) -> bool:
        for i, joint_id in enumerate(self._arm_joint_ids):
            low, high = self.model.jnt_range[joint_id]
            if q_arm[i] < low - 1e-6 or q_arm[i] > high + 1e-6:
                return True
        return False

    def solve(
        self,
        target_T_B_TCP: SE3,
        q_init: NDArray[np.floating[Any]] | None = None,
        *,
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
        max_iterations: int | None = None,
    ) -> IKResult:
        from ...core.transforms import geodesic_distance_rad

        target = np.asarray(target_T_B_TCP, dtype=np.float64).reshape(4, 4)
        T_B_E = target @ invert_transform(self.config.T_E_TCP_array)
        q_seed = (
            np.asarray(q_init, dtype=np.float64).reshape(-1)[: self.config.dof].copy()
            if q_init is not None
            else np.asarray(self.config.home_qpos, dtype=np.float64)
        )
        if q_seed.shape != (self.config.dof,):
            raise ValueError("xArm7 IK seed must contain exactly seven arm coordinates")
        gripper = (
            np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
            if gripper_qpos is not None
            else np.asarray(self.config.home_gripper_qpos, dtype=np.float64)
        )
        if gripper.shape != (self.config.gripper_dof,):
            raise ValueError("xArm7 gripper state must contain all six model hinge coordinates")

        config = self._set_configuration(q_seed, gripper)
        self.task.set_target(self._mink.SE3.from_matrix(T_B_E))
        iterations = int(max_iterations if max_iterations is not None else self.iterations)
        status = "max_iterations"
        it_used = 0
        error_msg: str | None = None
        for it in range(1, iterations + 1):
            it_used = it
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
            except Exception as exc:  # surfaced as an explicit solver result
                status = "error"
                error_msg = f"{type(exc).__name__}: {exc}"
                break
            config.integrate_inplace(velocity, dt=1e-2)
            error = self.task.compute_error(config)
            if float(np.linalg.norm(error[:3])) < self.tolerance and float(np.linalg.norm(error[3:])) < self.tolerance:
                status = "converged"
                break
        else:
            status = "max_iterations"

        q_solution = np.asarray(config.q[self._arm_addresses], dtype=np.float64).copy()
        T_B_E_fk = self._body_pose(q_solution, gripper, self.config.eef_body_name)
        T_B_TCP_fk = T_B_E_fk @ self.config.T_E_TCP_array
        pos_error = float(np.linalg.norm(T_B_TCP_fk[:3, 3] - target[:3, 3]))
        rot_error = geodesic_distance_rad(target[:3, :3], T_B_TCP_fk[:3, :3])
        limit_hit = self._limits_hit(q_solution)
        success = bool(pos_error <= 1e-4 and rot_error <= 1e-3 and status == "converged")
        return IKResult(
            success=success,
            qpos=q_solution,
            T_B_TCP_fk=T_B_TCP_fk,
            position_error_m=pos_error,
            rotation_error_rad=rot_error,
            iterations=it_used,
            status=status,
            status_code={"converged": 0, "max_iterations": 1, "no_solution": 2, "error": 3}.get(status, 9),
            joint_limit_hit=limit_hit,
            error=error_msg,
        )


class XArm7:
    """UFACTORY xArm7 arm and modeled parallel gripper."""

    robot_id = "xarm7"

    def __init__(
        self,
        mjcf_uri: str | Path | None = None,
        *,
        config: RobotConfig | None = None,
        ik: XArm7IKSolver | None = None,
        ik_settings: dict[str, Any] | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        path = Path(mjcf_uri) if mjcf_uri is not None else default_xarm7_xml()
        if not path.is_file():
            raise FileNotFoundError(f"xArm7 MJCF not found: {path}")
        self.mjcf_path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.config = config or build_xarm7_config(mjcf_uri=path)
        self.dof = self.config.dof
        self.max_reach_m = self.config.max_reach_m
        self.home_qpos = np.asarray(self.config.home_qpos, dtype=np.float64)
        self.home_gripper_qpos = np.asarray(self.config.home_gripper_qpos, dtype=np.float64)
        self._arm_joint_ids = _joint_ids(mujoco, self.model, self.config.joint_names)
        self._gripper_joint_ids = _joint_ids(mujoco, self.model, self.config.gripper_joint_names)
        self._arm_addresses = _qpos_addresses(self.model, self._arm_joint_ids)
        self._gripper_addresses = _qpos_addresses(self.model, self._gripper_joint_ids)
        # Calibrate around the actual named home pose; qpos0 is zero for the
        # pinned model, while the arm home is represented by its keyframe.
        home_all = _home_from_keyframe(mujoco, self.model)
        self._angles, self._widths = _gripper_calibration(
            mujoco, self.model, home_all, self._arm_joint_ids, self._gripper_joint_ids
        )

        settings = dict(ik_settings or {})
        self.ik_solver = ik or XArm7IKSolver(
            model=self.model,
            config=self.config,
            position_cost=float(settings.get("position_cost", 1.0)),
            orientation_cost=float(settings.get("orientation_cost", 1.0)),
            lm_damping=float(settings.get("lm_damping", 0.0)),
            iterations=int(settings.get("iterations", 100)),
            tolerance=float(settings.get("tolerance", 1e-5)),
            solver=str(settings.get("qp_backend", "quadprog")),
        )

    @property
    def max_gripper_width_m(self) -> float:
        return float(self._widths[0])

    @property
    def min_gripper_width_m(self) -> float:
        return float(self._widths[-1])

    def _set_qpos(
        self,
        q_arm: NDArray[np.floating[Any]],
        gripper: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        arm = np.array(q_arm, dtype=np.float64, copy=True).reshape(-1)
        if arm.shape != (self.dof,):
            raise ValueError("xArm7 arm qpos must contain exactly seven coordinates")
        grip = (
            self.home_gripper_qpos
            if gripper is None
            else np.array(gripper, dtype=np.float64, copy=True).reshape(-1)
        )
        if grip.shape != (self.config.gripper_dof,):
            raise ValueError("xArm7 gripper state must contain all six model hinge coordinates")
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self._arm_addresses] = arm
        self.data.qpos[self._gripper_addresses] = grip
        self._mujoco.mj_forward(self.model, self.data)

    def _body_pose(self, body_name: str) -> SE3:
        return _body_pose_from_data(self._mujoco, self.model, self.data, body_name)

    def gripper_width_to_qpos(self, width_m: float) -> NDArray[np.float64]:
        """Map physical pad opening to the six coupled hinge qpos values."""
        width = float(np.clip(width_m, self.min_gripper_width_m, self.max_gripper_width_m))
        angle = float(np.interp(width, self._widths[::-1], self._angles[::-1]))
        return np.full(self.config.gripper_dof, angle, dtype=np.float64)

    def qpos_to_gripper_width(self, gripper_qpos: NDArray[np.floating[Any]]) -> float:
        values = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
        if values.shape != (self.config.gripper_dof,):
            raise ValueError("xArm7 gripper state must contain all six model hinge coordinates")
        if not np.allclose(values, np.mean(values), atol=1e-5, rtol=0.0):
            raise ValueError("xArm7 gripper hinge qpos violate the source equality-coupled state")
        return float(np.interp(float(np.mean(values)), self._angles, self._widths))

    def forward_kinematics(
        self,
        qpos: NDArray[np.floating[Any]],
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> SE3:
        self._set_qpos(qpos, gripper_qpos)
        return self._body_pose(self.config.eef_body_name) @ self.config.T_E_TCP_array

    def solve_ik(
        self,
        target_T_B_TCP: SE3,
        q_init: NDArray[np.floating[Any]] | None = None,
        *,
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> IKResult:
        return self.ik_solver.solve(target_T_B_TCP, q_init, gripper_qpos=gripper_qpos)

    def check_collision(
        self,
        qpos: NDArray[np.floating[Any]],
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> CollisionResult:
        self._set_qpos(qpos, gripper_qpos)
        result = CollisionResult(has_self_collision=False)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body_ids = [int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])]
            names = [
                self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
                for body_id in body_ids
            ]
            pair = (names[0], names[1])
            result.max_penetration_m = max(result.max_penetration_m, float(contact.dist) * -1.0)
            if "world" in names:
                result.scene_contact_pairs.append(pair)
            elif frozenset(names) in _ADJACENT_PAIRS:
                result.adjacent_contact_pairs.append(pair)
            else:
                result.self_contact_pairs.append(pair)
        result.has_self_collision = bool(result.self_contact_pairs)
        if result.has_self_collision:
            result.code = 1
        elif result.scene_contact_pairs:
            result.code = 2
        return result

    def render(
        self,
        qpos: NDArray[np.floating[Any]],
        camera: Any,
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> RenderFrame:
        from ...vision.render import MujocoRenderer

        with MujocoRenderer(self, camera) as renderer:
            rendered = renderer.render(qpos, gripper_qpos=gripper_qpos)
        return RenderFrame(
            rgb=rendered["rgb"],
            robot_mask=rendered["robot_mask"],
            gripper_mask=rendered["gripper_mask"],
            depth_m=rendered["depth_m"],
            frame_index=int(rendered["frame_index"]),
            metadata={"segmentation": rendered["segmentation"], "camera_id": rendered["camera_id"]},
        )

    def body_positions(
        self,
        qpos: NDArray[np.floating[Any]],
        names: Sequence[str],
    ) -> dict[str, NDArray[np.float64]]:
        self._set_qpos(qpos)
        return {name: self._body_pose(name)[:3, 3] for name in names}

    def reach_ratio(self, T_B_TCP: SE3) -> float:
        return float(np.linalg.norm(np.asarray(T_B_TCP)[:3, 3])) / self.max_reach_m

    def apply_home(self) -> None:
        self._set_qpos(self.home_qpos, self.home_gripper_qpos)

    def set_state(
        self,
        qpos: NDArray[np.floating[Any]],
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        self._set_qpos(qpos, gripper_qpos)

    def body_pose(self, body_name: str) -> SE3:
        return self._body_pose(body_name)

    def tcp_pose(self) -> SE3:
        return self._body_pose(self.config.eef_body_name) @ self.config.T_E_TCP_array

    def kinematics_summary(self) -> dict[str, Any]:
        joints: dict[str, dict[str, Any]] = {}
        for name, jid in zip(self.config.joint_names, self._arm_joint_ids, strict=True):
            joints[name] = {
                "range": [float(v) for v in self.model.jnt_range[jid]],
                "qpos_adr": int(self.model.jnt_qposadr[jid]),
            }
        for name, jid in zip(self.config.gripper_joint_names, self._gripper_joint_ids, strict=True):
            joints[name] = {
                "range": [float(v) for v in self.model.jnt_range[jid]],
                "qpos_adr": int(self.model.jnt_qposadr[jid]),
            }
        return {
            "mjcf": str(self.mjcf_path),
            "mjcf_sha256": self.config.mjcf_sha256,
            "menagerie_commit": XARM7_MENAGERIE_COMMIT,
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "max_reach_m": self.max_reach_m,
            "gripper_width_range_m": [self.min_gripper_width_m, self.max_gripper_width_m],
            "eef_body": self.config.eef_body_name,
            "base_body": self.config.base_body_name,
            "T_E_TCP": self.config.T_E_TCP_array.tolist(),
            "joints": joints,
            "actuators": list(self.config.actuator_names),
            "bodies": [
                self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, i)
                for i in range(self.model.nbody)
            ],
        }
