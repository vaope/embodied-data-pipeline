"""MuJoCo adapter for the Trossen ViperX-300 6DOF morphology.

The MJCF is vendored from the official Google DeepMind MuJoCo Menagerie
``trossen_vx300s`` directory at commit
``c96a32d28fb5da84da38c1da4d749e7a13212855`` (BSD-3-Clause). The model's
``pinch`` site is the TCP origin. Its canonical orientation is derived from
the site approach axis and the left finger slide axis: local +x is approach,
local +z is the jaw axis, and local +y completes a right-handed frame.

The two slide joints are kinematically coupled by the MJCF equality
``right_finger = -left_finger``. Their valid model-space jaw gap is derived
from the paired finger-pad collision boxes at the joint limits. For the pinned
model this gives 0.015–0.087 m, with the affine mapping
``gap_m = 2 * left_finger_qpos - 0.027``. This is the simulated collision-pad
gap; the manufacturer's 42–116 mm gripper specification has a different
measurement convention and is not substituted here (see robot profile notes
and ADR-0019 owned by the project lead).

The manufacturer publishes 750 mm reach. This adapter records that source
value for reach normalization; it does not infer a different workspace from
the simplified MJCF geometry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ...core.schema import RobotConfig
from ...core.transforms import invert_transform, make_transform, matrix_from_quat_xyzw
from ..base import CollisionResult, IKResult, RenderFrame, SE3

__all__ = [
    "VIPERX300_EEF_BODY_NAME",
    "VIPERX300_GRIPPER_JOINT_NAMES",
    "VIPERX300_JOINT_NAMES",
    "VIPERX300_MAX_REACH_M",
    "VIPERX300_TCP_SITE_NAME",
    "ViperX300",
    "ViperX300IKSolver",
    "build_viperx_config",
    "default_viperx_xml",
]

VIPERX300_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
VIPERX300_GRIPPER_JOINT_NAMES = ["left_finger", "right_finger"]
VIPERX300_EEF_BODY_NAME = "gripper_link"
VIPERX300_BASE_BODY_NAME = "base_link"
VIPERX300_TCP_SITE_NAME = "pinch"

# Manufacturer's published reach for the ViperX-300 6DOF (Trossen Interbotix
# specification). The Menagerie MJCF does not declare a reach scalar.
VIPERX300_MAX_REACH_M = 0.750

MENAGERIE_COMMIT = "c96a32d28fb5da84da38c1da4d749e7a13212855"
MENAGERIE_SOURCE = (
    "google-deepmind/mujoco_menagerie@"
    f"{MENAGERIE_COMMIT}:trossen_vx300s/vx300s.xml"
)
TROSSEN_SPECS_URL = (
    "https://docs.trossenrobotics.com/interbotix_xsarms_docs/"
    "specifications/vx300s.html"
)

_ADJACENT_PAIRS = {
    frozenset({"base_link", "shoulder_link"}),
    frozenset({"shoulder_link", "upper_arm_link"}),
    frozenset({"upper_arm_link", "upper_forearm_link"}),
    frozenset({"upper_forearm_link", "lower_forearm_link"}),
    frozenset({"lower_forearm_link", "wrist_link"}),
    frozenset({"wrist_link", "gripper_link"}),
    frozenset({"gripper_link", "gripper_prop_link"}),
    frozenset({"gripper_link", "left_finger_link"}),
    frozenset({"gripper_link", "right_finger_link"}),
}

_MESH_DIR = Path(__file__).resolve().parents[3].parent / "assets" / "mjcf" / "trossen_vx300s"


def default_viperx_xml(repo_root: str | Path | None = None) -> Path:
    """Locate the pinned ViperX-300 MJCF in the vendored asset tree."""
    if repo_root is not None:
        candidate = Path(repo_root) / "assets" / "mjcf" / "trossen_vx300s" / "vx300s.xml"
        if candidate.is_file():
            return candidate
    if _MESH_DIR.joinpath("vx300s.xml").is_file():
        return _MESH_DIR / "vx300s.xml"
    raise FileNotFoundError(
        "cannot find assets/mjcf/trossen_vx300s/vx300s.xml; run "
        "`python scripts/fetch_viperx300.py` or pass an explicit mjcf_uri"
    )


def _mujoco_name_id(model: Any, object_type: Any, name: str) -> int:
    import mujoco

    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"required ViperX model element {name!r} is missing")
    return object_id


def _joint_address_map(model: Any) -> dict[str, tuple[int, int]]:
    """Return named scalar-joint qpos/dof addresses and reject unsupported types."""
    import mujoco

    result: dict[str, tuple[int, int]] = {}
    for name in [*VIPERX300_JOINT_NAMES, *VIPERX300_GRIPPER_JOINT_NAMES]:
        jid = _mujoco_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE if name not in VIPERX300_GRIPPER_JOINT_NAMES else mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"joint {name!r} has an unsupported MJCF type")
        result[name] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
    qpos_addresses = [result[name][0] for name in [*VIPERX300_JOINT_NAMES, *VIPERX300_GRIPPER_JOINT_NAMES]]
    if qpos_addresses != list(range(8)):
        raise ValueError(f"ViperX joint ordering changed; expected qpos addresses 0..7, got {qpos_addresses}")
    return result


def _read_keyframe_qpos(model: Any, key_name: str, joint_names: Sequence[str]) -> NDArray[np.float64]:
    import mujoco

    key_id = _mujoco_name_id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    addresses = _joint_address_map(model)
    return np.asarray(
        [model.key_qpos[key_id, addresses[name][0]] for name in joint_names], dtype=np.float64
    )


def _validate_mimic_constraint(model: Any, address: dict[str, tuple[int, int]]) -> None:
    """Require the Menagerie left/right finger equality, rather than assume it."""
    import mujoco

    left_id = _mujoco_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")
    right_id = _mujoco_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger")
    expected_type = int(mujoco.mjtEq.mjEQ_JOINT)
    matches: list[int] = []
    for eq_id in range(model.neq):
        if (
            int(model.eq_type[eq_id]) == expected_type
            and int(model.eq_obj1id[eq_id]) == left_id
            and int(model.eq_obj2id[eq_id]) == right_id
        ):
            matches.append(eq_id)
    if len(matches) != 1:
        raise ValueError("MJCF must contain one left_finger -> right_finger joint equality")
    coeffs = np.asarray(model.eq_data[matches[0], :5], dtype=np.float64)
    if not np.allclose(coeffs, [0.0, -1.0, 0.0, 0.0, 0.0], atol=1e-12, rtol=0.0):
        raise ValueError(f"unexpected finger mimic polynomial: {coeffs.tolist()}")
    left_lo, left_hi = np.asarray(model.jnt_range[left_id], dtype=np.float64)
    right_lo, right_hi = np.asarray(model.jnt_range[right_id], dtype=np.float64)
    if not np.allclose([right_lo, right_hi], [-left_hi, -left_lo], atol=1e-12, rtol=0.0):
        raise ValueError("finger joint ranges are not mirrored around zero")
    # Keep the map argument used so accidental address mismatch is caught here.
    if address["right_finger"][0] != int(model.jnt_qposadr[right_id]):
        raise ValueError("right_finger qpos address mismatch")


def _pad_gap_at_q(model: Any, q_arm_home: NDArray[np.float64], q_left: float) -> float:
    """Measure paired collision-pad separation along the current jaw axis."""
    import mujoco

    address = _joint_address_map(model)
    _validate_mimic_constraint(model, address)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for name, value in zip(VIPERX300_JOINT_NAMES, q_arm_home, strict=True):
        data.qpos[address[name][0]] = float(value)
    data.qpos[address["left_finger"][0]] = float(q_left)
    data.qpos[address["right_finger"][0]] = -float(q_left)
    mujoco.mj_forward(model, data)

    left_joint = _mujoco_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")
    jaw_axis_B = np.asarray(data.xaxis[left_joint], dtype=np.float64)
    jaw_axis_B /= np.linalg.norm(jaw_axis_B)
    pad_ids: dict[str, dict[int, int]] = {"left": {}, "right": {}}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        for side in ("left", "right"):
            prefix = f"{side}_finger_pad_"
            if name.startswith(prefix):
                suffix = name[len(prefix) :]
                if suffix.isdigit():
                    pad_ids[side][int(suffix)] = geom_id
    if not pad_ids["left"] or set(pad_ids["left"]) != set(pad_ids["right"]):
        raise ValueError("paired left/right finger-pad collision geoms are incomplete")

    gaps: list[float] = []
    for suffix in sorted(pad_ids["left"]):
        centers: list[float] = []
        radii: list[float] = []
        for side in ("left", "right"):
            geom_id = pad_ids[side][suffix]
            center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            half_sizes = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            projection_radius = sum(
                abs(float(np.dot(jaw_axis_B, rotation[:, axis]))) * float(half_sizes[axis])
                for axis in range(3)
            )
            centers.append(float(np.dot(center, jaw_axis_B)))
            radii.append(projection_radius)
        gaps.append(abs(centers[0] - centers[1]) - radii[0] - radii[1])
    if not np.all(np.isfinite(gaps)) or min(gaps) <= 0.0:
        raise ValueError(f"invalid pad gap geometry at left_finger qpos={q_left}: {gaps}")
    if max(gaps) - min(gaps) > 1e-6:
        raise ValueError(f"finger pad pairs disagree on jaw gap by more than 1e-6 m: {gaps}")
    return float(np.mean(gaps))


def _derive_model_contract(model: Any) -> dict[str, Any]:
    import mujoco

    address = _joint_address_map(model)
    _validate_mimic_constraint(model, address)
    if int(model.nq) != 8 or int(model.nv) != 8:
        raise ValueError(f"unexpected ViperX state dimensions nq={model.nq}, nv={model.nv}")
    eef_body = _mujoco_name_id(model, mujoco.mjtObj.mjOBJ_BODY, VIPERX300_EEF_BODY_NAME)
    base_body = _mujoco_name_id(model, mujoco.mjtObj.mjOBJ_BODY, VIPERX300_BASE_BODY_NAME)
    site_id = _mujoco_name_id(model, mujoco.mjtObj.mjOBJ_SITE, VIPERX300_TCP_SITE_NAME)
    if int(model.site_bodyid[site_id]) != eef_body:
        raise ValueError("pinch TCP site must be attached directly to gripper_link")

    q_home = _read_keyframe_qpos(model, "home", VIPERX300_JOINT_NAMES)
    q_gripper_home = _read_keyframe_qpos(model, "home", VIPERX300_GRIPPER_JOINT_NAMES)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for name, value in zip(VIPERX300_JOINT_NAMES, q_home, strict=True):
        data.qpos[address[name][0]] = float(value)
    for name, value in zip(VIPERX300_GRIPPER_JOINT_NAMES, q_gripper_home, strict=True):
        data.qpos[address[name][0]] = float(value)
    mujoco.mj_forward(model, data)
    R_B_E = np.asarray(data.xmat[eef_body], dtype=np.float64).reshape(3, 3)
    left_jid = _mujoco_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")
    jaw_z_E = R_B_E.T @ np.asarray(data.xaxis[left_jid], dtype=np.float64)
    jaw_z_E /= np.linalg.norm(jaw_z_E)

    # MuJoCo stores site quaternions in wxyz order; project boundaries here are matrices.
    site_quat_wxyz = np.asarray(model.site_quat[site_id], dtype=np.float64)
    R_E_site = matrix_from_quat_xyzw(site_quat_wxyz[[1, 2, 3, 0]])
    tcp_x_E = R_E_site[:, 0]
    jaw_z_E = jaw_z_E - tcp_x_E * float(np.dot(jaw_z_E, tcp_x_E))
    norm_z = float(np.linalg.norm(jaw_z_E))
    if norm_z < 1e-8:
        raise ValueError("pinch approach axis is parallel to the finger slide axis")
    jaw_z_E /= norm_z
    tcp_y_E = np.cross(jaw_z_E, tcp_x_E)
    tcp_y_E /= np.linalg.norm(tcp_y_E)
    tcp_x_E = np.cross(tcp_y_E, jaw_z_E)
    R_E_TCP = np.column_stack([tcp_x_E, tcp_y_E, jaw_z_E])
    if not np.allclose(R_E_TCP.T @ R_E_TCP, np.eye(3), atol=1e-9) or np.linalg.det(R_E_TCP) < 0.999999:
        raise ValueError("model-derived ViperX TCP orientation is not a right-handed rotation")
    T_E_TCP = make_transform(R_E_TCP, model.site_pos[site_id])

    left_jid = _mujoco_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")
    q_lo, q_hi = [float(v) for v in model.jnt_range[left_jid]]
    width_lo = _pad_gap_at_q(model, q_home, q_lo)
    width_hi = _pad_gap_at_q(model, q_home, q_hi)
    if width_hi <= width_lo:
        raise ValueError(f"jaw opening is not monotonic across joint limits: {width_lo}, {width_hi}")

    return {
        "address": address,
        "base_body_name": VIPERX300_BASE_BODY_NAME,
        "eef_body_name": VIPERX300_EEF_BODY_NAME,
        "gripper_geom_names": [
            "gripper_link",
            "gripper_prop_link",
            "left_finger_link",
            "right_finger_link",
        ],
        "gripper_home": q_gripper_home,
        "gripper_joint_names": list(VIPERX300_GRIPPER_JOINT_NAMES),
        "gripper_limits": [
            [float(v) for v in model.jnt_range[_mujoco_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]]
            for n in VIPERX300_GRIPPER_JOINT_NAMES
        ],
        "home_qpos": q_home,
        "joint_limits": [
            [float(v) for v in model.jnt_range[_mujoco_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]]
            for n in VIPERX300_JOINT_NAMES
        ],
        "joint_names": list(VIPERX300_JOINT_NAMES),
        "max_gripper_width_m": width_hi,
        "min_gripper_width_m": width_lo,
        "site_id": site_id,
        "T_E_TCP": T_E_TCP,
    }


def build_viperx_config(
    *,
    mjcf_uri: str | Path,
    robot_id: str = "viperx300",
    profile: Any | None = None,
) -> RobotConfig:
    """Build a profile checked against named joints, keyframe and pad geometry."""
    import mujoco

    path = Path(mjcf_uri)
    if not path.is_file():
        raise FileNotFoundError(f"ViperX MJCF not found: {path}")
    model = mujoco.MjModel.from_xml_path(str(path))
    derived = _derive_model_contract(model)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = {
        "dof": len(derived["joint_names"]),
        "gripper_dof": len(derived["gripper_joint_names"]),
        "max_reach_m": VIPERX300_MAX_REACH_M,
        "eef_body_name": derived["eef_body_name"],
        "base_body_name": derived["base_body_name"],
        "joint_names": derived["joint_names"],
        "gripper_joint_names": derived["gripper_joint_names"],
        "gripper_width_min_m": derived["min_gripper_width_m"],
        "gripper_width_max_m": derived["max_gripper_width_m"],
    }
    if profile is not None:
        mismatches: list[str] = []
        for field, value in contract.items():
            declared = getattr(profile, field, None)
            if declared is None or declared == [] or declared == "":
                continue
            if isinstance(value, float):
                matches = bool(np.isclose(float(declared), value, atol=1e-9, rtol=0.0))
            else:
                matches = declared == value
            if not matches:
                mismatches.append(f"{field}: profile says {declared!r}, model/source gives {value!r}")
        profile_license = str(getattr(profile, "mjcf_license", "") or "")
        if profile_license and profile_license != "BSD-3-Clause":
            mismatches.append(f"mjcf_license: profile says {profile_license!r}, source gives 'BSD-3-Clause'")
        if mismatches:
            raise ValueError("robot profile disagrees with the ViperX model contract: " + "; ".join(mismatches))

    actuator_names = []
    for actuator_id in range(model.nu):
        import mujoco

        actuator_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or "")
    return RobotConfig(
        robot_id=robot_id or (getattr(profile, "robot_id", None) or "viperx300"),
        dof=contract["dof"],
        gripper_dof=contract["gripper_dof"],
        max_reach_m=contract["max_reach_m"],
        home_qpos=derived["home_qpos"].tolist(),
        home_gripper_qpos=derived["gripper_home"].tolist(),
        mjcf_uri=str(path),
        mjcf_sha256=sha,
        mjcf_license="BSD-3-Clause",
        mjcf_source=MENAGERIE_SOURCE,
        T_E_TCP=derived["T_E_TCP"].tolist(),
        joint_names=derived["joint_names"],
        gripper_joint_names=derived["gripper_joint_names"],
        actuator_names=actuator_names,
        position_limits=derived["joint_limits"],
        eef_body_name=derived["eef_body_name"],
        base_body_name=derived["base_body_name"],
        gripper_geom_names=derived["gripper_geom_names"],
        max_joint_jump_rad=float(getattr(getattr(profile, "ik", None), "max_joint_jump_rad", 1.5)),
        notes=[
            f"Reach 0.750 m is the manufacturer-published value: {TROSSEN_SPECS_URL}",
            "Simulated jaw gap is derived from paired Menagerie finger-pad collision boxes at slide limits: "
            f"{derived['min_gripper_width_m']:.6f}–{derived['max_gripper_width_m']:.6f} m.",
            "Manufacturer specification lists a 42–116 mm gripper range; that is a different measurement "
            "convention than this MJCF's collision-pad gap and is not used for simulated qpos mapping.",
            "Model equality couples right_finger = -left_finger; persisted gripper_qpos retains both slide values.",
            "TCP orientation: approach is the pinch site's +x axis; jaw is the left_finger slide axis; "
            "the middle axis is their right-handed completion.",
        ],
    )


@dataclass
class ViperX300IKSolver:
    """Mink IK for a TCP target, controlling the six named arm joints."""

    model: Any
    config: RobotConfig
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    lm_damping: float = 0.0
    iterations: int = 100
    tolerance: float = 1e-5
    solver: str = "quadprog"

    def __post_init__(self) -> None:
        import mujoco
        import mink

        self._mujoco = mujoco
        self._mink = mink
        self._addresses = _joint_address_map(self.model)
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

    def _set_configuration(
        self,
        q_arm: NDArray[np.float64],
        gripper: NDArray[np.float64],
    ) -> Any:
        qpos = self._configuration.q.copy()
        for name, value in zip(self.config.joint_names, q_arm, strict=True):
            qpos[self._addresses[name][0]] = value
        for name, value in zip(self.config.gripper_joint_names, gripper, strict=True):
            qpos[self._addresses[name][0]] = value
        self._configuration.update(qpos)
        return self._configuration

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
        target_T_B_E = target @ invert_transform(self.config.T_E_TCP_array)
        q_seed = (
            np.asarray(q_init, dtype=np.float64).reshape(-1)[: self.config.dof].copy()
            if q_init is not None
            else self._q_arm.copy()
        )
        if q_seed.shape != (self.config.dof,):
            raise ValueError(f"q_init must contain {self.config.dof} arm coordinates")
        gripper = (
            np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
            if gripper_qpos is not None
            else np.asarray(self.config.home_gripper_qpos, dtype=np.float64)
        )
        if gripper.shape != (self.config.gripper_dof,):
            raise ValueError(f"gripper_qpos must contain {self.config.gripper_dof} coordinates")
        configuration = self._set_configuration(q_seed, gripper)
        self.task.set_target(self._mink.SE3.from_matrix(target_T_B_E))

        n_iterations = int(max_iterations if max_iterations is not None else self.iterations)
        status = "max_iterations"
        error_msg: str | None = None
        it_used = 0
        for it in range(1, n_iterations + 1):
            it_used = it
            try:
                velocity = self._mink.solve_ik(
                    configuration,
                    [self.task],
                    dt=1e-2,
                    solver=self.solver,
                    damping=1e-5,
                    limits=self.limits,
                )
            except self._mink.NoSolutionFound:
                status = "no_solution"
                break
            except Exception as exc:  # surfaced to the trajectory manifest
                status = "error"
                error_msg = f"{type(exc).__name__}: {exc}"
                break
            configuration.integrate_inplace(velocity, dt=1e-2)
            task_error = self.task.compute_error(configuration)
            if float(np.linalg.norm(task_error[:3])) < self.tolerance and float(
                np.linalg.norm(task_error[3:])
            ) < self.tolerance:
                status = "converged"
                break

        q_solution = np.asarray(
            [configuration.q[self._addresses[name][0]] for name in self.config.joint_names],
            dtype=np.float64,
        )
        T_B_E_fk = self._body_pose(q_solution, gripper, self.config.eef_body_name)
        T_B_TCP_fk = T_B_E_fk @ self.config.T_E_TCP_array
        position_error = float(np.linalg.norm(T_B_TCP_fk[:3, 3] - target[:3, 3]))
        rotation_error = geodesic_distance_rad(target[:3, :3], T_B_TCP_fk[:3, :3])
        limit_hit = self._limits_hit(q_solution)
        success = bool(position_error <= 1e-4 and rotation_error <= 1e-3 and status == "converged")
        return IKResult(
            success=success,
            qpos=q_solution,
            T_B_TCP_fk=T_B_TCP_fk,
            position_error_m=position_error,
            rotation_error_rad=rotation_error,
            iterations=it_used,
            status=status,
            status_code={"converged": 0, "max_iterations": 1, "no_solution": 2, "error": 3}.get(status, 9),
            joint_limit_hit=limit_hit,
            error=error_msg,
        )

    def _limits_hit(self, q_arm: NDArray[np.float64]) -> bool:
        for name, value in zip(self.config.joint_names, q_arm, strict=True):
            jid = _mujoco_name_id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = self.model.jnt_range[jid]
            if value < lo - 1e-6 or value > hi + 1e-6:
                return True
        return False

    def _body_pose(
        self,
        q_arm: NDArray[np.float64],
        gripper: NDArray[np.float64],
        body_name: str,
    ) -> SE3:
        import mujoco

        data = mujoco.MjData(self.model)
        data.qpos[:] = self.model.qpos0
        for name, value in zip(self.config.joint_names, q_arm, strict=True):
            data.qpos[self._addresses[name][0]] = value
        for name, value in zip(self.config.gripper_joint_names, gripper, strict=True):
            data.qpos[self._addresses[name][0]] = value
        mujoco.mj_forward(self.model, data)
        bid = _mujoco_name_id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        from ...core.transforms import make_transform

        return make_transform(data.xmat[bid].reshape(3, 3), data.xpos[bid])


class ViperX300:
    """ViperX-300 arm and coupled two-finger gripper on the pinned MJCF."""

    robot_id = "viperx300"

    def __init__(
        self,
        mjcf_uri: str | Path | None = None,
        *,
        config: RobotConfig | None = None,
        ik: ViperX300IKSolver | None = None,
        ik_settings: dict[str, Any] | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        path = Path(mjcf_uri) if mjcf_uri is not None else default_viperx_xml()
        if not path.is_file():
            raise FileNotFoundError(f"ViperX MJCF not found: {path}")
        self.mjcf_path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.config = config or build_viperx_config(mjcf_uri=path)
        if self.config.robot_id != self.robot_id:
            raise ValueError(f"ViperX adapter requires robot_id={self.robot_id!r}, got {self.config.robot_id!r}")
        if self.config.mjcf_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != self.config.mjcf_sha256:
            raise ValueError("ViperX MJCF SHA-256 differs from RobotConfig")

        self.dof = self.config.dof
        self.max_reach_m = self.config.max_reach_m
        self.home_qpos = np.asarray(self.config.home_qpos, dtype=np.float64)
        self.home_gripper_qpos = np.asarray(self.config.home_gripper_qpos, dtype=np.float64)
        self._addresses = _joint_address_map(self.model)
        contract = _derive_model_contract(self.model)
        self.min_gripper_width_m = float(contract["min_gripper_width_m"])
        self.max_gripper_width_m = float(contract["max_gripper_width_m"])
        if not np.isclose(self.config.T_E_TCP_array, contract["T_E_TCP"], atol=1e-9).all():
            raise ValueError("RobotConfig T_E_TCP differs from the model-derived pinch grasp frame")

        settings = dict(ik_settings or {})
        self.ik_solver = ik or ViperX300IKSolver(
            model=self.model,
            config=self.config,
            position_cost=float(settings.get("position_cost", 1.0)),
            orientation_cost=float(settings.get("orientation_cost", 1.0)),
            lm_damping=float(settings.get("lm_damping", 0.0)),
            iterations=int(settings.get("iterations", 100)),
            tolerance=float(settings.get("tolerance", 1e-5)),
            solver=str(settings.get("qp_backend", "quadprog")),
        )

    def _set_qpos(
        self,
        q_arm: NDArray[np.floating[Any]],
        gripper: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        q = np.array(q_arm, dtype=np.float64, copy=True).reshape(-1)
        if q.shape != (self.dof,):
            raise ValueError(f"qpos must contain {self.dof} arm coordinates")
        g = np.array(
            self.home_gripper_qpos if gripper is None else gripper,
            dtype=np.float64,
            copy=True,
        ).reshape(-1)
        if g.shape != (self.config.gripper_dof,):
            raise ValueError(f"gripper_qpos must contain {self.config.gripper_dof} coordinates")
        self.data.qpos[:] = self.model.qpos0
        for name, value in zip(self.config.joint_names, q, strict=True):
            self.data.qpos[self._addresses[name][0]] = value
        for name, value in zip(self.config.gripper_joint_names, g, strict=True):
            self.data.qpos[self._addresses[name][0]] = value
        self._mujoco.mj_forward(self.model, self.data)

    def _body_pose(self, body_name: str) -> SE3:
        bid = _mujoco_name_id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_name)
        return make_transform(self.data.xmat[bid].reshape(3, 3), self.data.xpos[bid])

    def gripper_width_to_qpos(self, width_m: float) -> NDArray[np.float64]:
        """Clamp a canonical width to the simulated pad-gap range and return both finger joints.

        The inverse map uses the geometry-derived endpoint calibration. For the
        pinned Menagerie model this reduces to ``left_q = (width_m + 0.027)/2``
        and ``right_q = -left_q``. Values outside 15–87 mm clamp deterministically
        to the nearest model limit; the robot stage should separately record the
        pre-clamp requested width and count clipped frames.
        """
        width = float(width_m)
        if not np.isfinite(width):
            raise ValueError("gripper width must be finite")
        width_clamped = float(np.clip(width, self.min_gripper_width_m, self.max_gripper_width_m))
        left_lo, left_hi = self.model.jnt_range[
            _mujoco_name_id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, "left_finger")
        ]
        fraction = (width_clamped - self.min_gripper_width_m) / (
            self.max_gripper_width_m - self.min_gripper_width_m
        )
        left_q = float(left_lo + fraction * (left_hi - left_lo))
        return np.array([left_q, -left_q], dtype=np.float64)

    def qpos_to_gripper_width(self, gripper_qpos: NDArray[np.floating[Any]]) -> float:
        """Map mirrored finger slide positions to their simulated collision-pad gap in metres."""
        q = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
        if q.shape != (2,):
            raise ValueError("gripper_qpos must be [left_finger, right_finger]")
        if not np.all(np.isfinite(q)):
            raise ValueError("gripper_qpos must be finite")
        if not np.isclose(q[1], -q[0], atol=1e-8, rtol=0.0):
            raise ValueError("left_finger/right_finger qpos violate the MJCF mimic equality")
        left_id = _mujoco_name_id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, "left_finger")
        left_lo, left_hi = [float(v) for v in self.model.jnt_range[left_id]]
        fraction = (float(q[0]) - left_lo) / (left_hi - left_lo)
        return float(self.min_gripper_width_m + fraction * (self.max_gripper_width_m - self.min_gripper_width_m))

    def forward_kinematics(
        self,
        qpos: NDArray[np.floating[Any]],
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> SE3:
        """Return ``T_B_TCP`` using the model-derived pinch grasp frame."""
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
        """Classify MuJoCo contacts as adjacent, self, or world/scene contact."""
        self._set_qpos(qpos, gripper_qpos)
        mujoco = self._mujoco
        result = CollisionResult(has_self_collision=False)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body_names: list[str] = []
            for geom_id in (int(contact.geom1), int(contact.geom2)):
                body_id = int(self.model.geom_bodyid[geom_id])
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                body_names.append(name or "world")
            body_a, body_b = body_names
            pair = (body_a, body_b)
            result.max_penetration_m = max(result.max_penetration_m, max(0.0, -float(contact.dist)))
            if body_a == "world" or body_b == "world":
                result.scene_contact_pairs.append(pair)
            elif frozenset(pair) in _ADJACENT_PAIRS:
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
        *,
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> RenderFrame:
        """Render through the shared MuJoCo renderer and return the typed frame contract."""
        from ...vision.render import MujocoRenderer

        with MujocoRenderer(self, camera) as renderer:
            output = renderer.render(qpos, gripper_qpos=gripper_qpos)
        return RenderFrame(
            rgb=np.asarray(output["rgb"], dtype=np.uint8),
            robot_mask=np.asarray(output["robot_mask"], dtype=bool),
            gripper_mask=np.asarray(output["gripper_mask"], dtype=bool),
            depth_m=np.asarray(output["depth_m"], dtype=np.float32),
            frame_index=int(output["frame_index"]),
            metadata={"camera_id": int(output["camera_id"]), "mjcf_source": MENAGERIE_SOURCE},
        )

    def body_positions(
        self, qpos: NDArray[np.floating[Any]], names: Sequence[str]
    ) -> dict[str, NDArray[np.float64]]:
        self._set_qpos(qpos)
        return {name: self._body_pose(name)[:3, 3] for name in names}

    def reach_ratio(self, T_B_TCP: SE3) -> float:
        return float(np.linalg.norm(np.asarray(T_B_TCP, dtype=np.float64)[:3, 3])) / self.max_reach_m

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
        joints = {}
        for name in [*self.config.joint_names, *self.config.gripper_joint_names]:
            jid = _mujoco_name_id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            joints[name] = {
                "range": [float(v) for v in self.model.jnt_range[jid]],
                "qpos_adr": int(self.model.jnt_qposadr[jid]),
                "dof_adr": int(self.model.jnt_dofadr[jid]),
            }
        return {
            "mjcf": str(self.mjcf_path),
            "mjcf_sha256": self.config.mjcf_sha256,
            "mjcf_source": MENAGERIE_SOURCE,
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "max_reach_m": self.max_reach_m,
            "min_gripper_width_m": self.min_gripper_width_m,
            "max_gripper_width_m": self.max_gripper_width_m,
            "eef_body": self.config.eef_body_name,
            "tcp_site": VIPERX300_TCP_SITE_NAME,
            "T_E_TCP": self.config.T_E_TCP_array.tolist(),
            "joints": joints,
        }
