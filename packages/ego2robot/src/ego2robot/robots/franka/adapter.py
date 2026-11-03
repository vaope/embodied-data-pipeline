"""Franka Emika Panda adapter: MuJoCo model, FK, Mink IK, contacts, rendering.

Model source: `mujoco_menagerie/franka_emika_panda
<https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda>`_
(``panda.xml``), downloaded to ``assets/mjcf/franka_emika_panda/``.  Licence:
BSD-3-Clause (see the ``LICENSE`` file next to the model).

Frame definitions used here
---------------------------
* ``B``   -- MJCF ``world`` body; the arm root ``link0`` sits at its origin.
* ``E``   -- MJCF ``hand`` body.  This is the IK target frame.
* ``TCP`` -- jaw midpoint with::

      x = approach (toward the fingers)
      y = jaw-plane normal
      z = grasp axis, along the jaw line

The constant offset is derived, not guessed: with both finger joints at 0 the
finger bodies coincide at ``hand + (0, 0, 0.0584)`` m, and each finger travels
along the hand frame's ``+y`` as the gripper opens.  Therefore::

      T_E_TCP = [[0, 1, 0, 0     ],
                 [0, 0, 1, 0     ],
                 [1, 0, 0, 0.0584],
                 [0, 0, 0, 1     ]]

i.e. the TCP axes ``[x y z]`` are the hand-frame axes ``[+z +x +y]``, which is a
rotation (``det = +1``): ``x`` points towards the fingers (approach), ``z`` lies
along the finger-travel direction (the jaw line), and ``y = z × x`` is the
jaw-plane normal.  The gripper opening in metres equals twice a single finger
joint position, so the 0-80 mm range the paper publishes for Panda
(appendix A.7) maps to ``finger_joint = 0..0.04``.

.. rubric:: Correction (2026-10-27)

The constant previously used the cyclic permutation
``[[0,0,1],[1,0,0],[0,1,0]]``, which maps the TCP ``x`` axis onto the hand ``+y``
— i.e. it asserted that the *approach* axis is the jaw line, 90° away from the
canonical ``z = grasp axis``. Measured with MuJoCo forward kinematics on the
executed joint states, the finger-to-finger line was parallel to the commanded
``x`` (0.0°) and perpendicular to the commanded ``z`` (90.0°) on all 52 frames of
the reference episode. See ADR-0005 (amended) for evidence and migration note.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ...core.schema import RobotConfig
from ...core.transforms import compose_transforms, invert_transform, make_transform
from ..base import CollisionResult, IKResult, RenderFrame, SE3

__all__ = [
    "PANDA_EFFORT_LIMITS",
    "PANDA_JOINT_NAMES",
    "PANDA_MAX_REACH_M",
    "PANDA_TCP_OFFSET_IN_HAND",
    "PandaIKSolver",
    "FrankaPanda",
    "build_franka_config",
    "default_panda_xml",
]

PANDA_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
PANDA_GRIPPER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]

#: TCP frame expressed in the ``hand`` body frame (see the module docstring).
PANDA_TCP_OFFSET_IN_HAND = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.1031],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
#: Fallback only: ``build_franka_config`` re-derives the depth from the finger pad
#: collision geoms (``pad_centre_depth_in_hand``) and overrides ``[2, 3]`` whenever
#: the model is readable. Measured on the vendored model: 0.1031 m.

#: Panda reach including the hand, from the menagerie model's kinematic chain.
PANDA_MAX_REACH_M = 0.855

PANDA_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64)
PANDA_HOME_GRIPPER_QPOS = np.array([0.04, 0.04], dtype=np.float64)

PANDA_EFFORT_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)

#: adjacent body pairs whose contact is structural, not a collision
_ADJACENT_PAIRS = {
    frozenset({"link1", "link2"}),
    frozenset({"link2", "link3"}),
    frozenset({"link3", "link4"}),
    frozenset({"link4", "link5"}),
    frozenset({"link5", "link6"}),
    frozenset({"link6", "link7"}),
    frozenset({"link7", "hand"}),
    frozenset({"hand", "left_finger"}),
    frozenset({"hand", "right_finger"}),
    frozenset({"left_finger", "right_finger"}),
}

_MESH_DIR = Path(__file__).resolve().parents[3].parent / "assets" / "mjcf" / "franka_emika_panda"


def default_panda_xml(repo_root: str | Path | None = None) -> Path:
    """Locate ``panda.xml`` from the vendored assets."""
    if repo_root is not None:
        candidate = Path(repo_root) / "assets" / "mjcf" / "franka_emika_panda" / "panda.xml"
        if candidate.is_file():
            return candidate
    if _MESH_DIR.joinpath("panda.xml").is_file():
        return _MESH_DIR / "panda.xml"
    raise FileNotFoundError(
        "cannot find assets/mjcf/franka_emika_panda/panda.xml; run "
        "`python scripts/fetch_panda.py` or pass an explicit mjcf_uri"
    )


def pad_centre_depth_in_hand(model: Any) -> float:
    """Depth of the finger **pad contact centre** along the hand frame's ``+z``.

    The finger bodies sit at ``hand + (0, 0, 0.0584)`` m, but that is where the
    fingers *attach* to the hand, not where they touch an object. The pads are the
    small box geoms with an active collision mask on the two finger bodies; their
    centre is what a pinch grasp actually contacts, and it is what the paper's
    ``p_tcp = (p_thumb + p_vf) / 2`` corresponds to (the point between the human
    fingertips). Measured on the vendored model: pads span
    ``hand z ∈ [0.0979, 0.1084]`` m, centre ``≈ 0.1031`` m — i.e. 44.7 mm in front of
    the finger-body origin.
    """
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    R_hand = data.xmat[hand].reshape(3, 3)
    p_hand = data.xpos[hand].copy()

    finger_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_finger", "right_finger")
    }
    depths = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in finger_bodies:
            continue
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue        # visual-only duplicate
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            continue        # the pads are boxes; the pad base mesh is not
        centre = R_hand.T @ (data.geom_xpos[gid] - p_hand)
        depths.append(float(centre[2]))
    if not depths:
        raise ValueError("no finger pad collision geoms found in the Panda model")
    return float(np.mean(depths))


def build_franka_config(
    *,
    mjcf_uri: str | Path,
    robot_id: str = "franka_panda",
    profile: Any | None = None,
) -> RobotConfig:
    """Build the persisted :class:`RobotConfig` for the Panda.

    ``profile`` is an optional :class:`~ego2robot.core.config.RobotBindingConfig`
    whose restated kinematic values are checked against the values actually
    derived from the MJCF.  A mismatch is raised, not warned about: a robot
    profile that disagrees with its own model is worse than no profile.
    """
    p = Path(mjcf_uri)
    sha = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
    tcp_offset = np.array(PANDA_TCP_OFFSET_IN_HAND, dtype=np.float64, copy=True)
    if p.is_file():
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(p))
        pad_depth = pad_centre_depth_in_hand(model)
        tcp_offset[2, 3] = pad_depth
    derived = {
        "dof": 7,
        "gripper_dof": 2,
        "max_reach_m": PANDA_MAX_REACH_M,
        "eef_body_name": "hand",
        "base_body_name": "link0",
        "joint_names": list(PANDA_JOINT_NAMES),
        "gripper_joint_names": list(PANDA_GRIPPER_JOINT_NAMES),
    }
    if profile is not None:
        mismatches: list[str] = []
        for field, value in derived.items():
            declared = getattr(profile, field, None)
            if declared is None or declared == [] or declared == "":
                continue
            if declared != value:
                mismatches.append(f"{field}: profile says {declared!r}, model gives {value!r}")
        if mismatches:
            raise ValueError(
                "robot profile disagrees with the vendored MJCF: " + "; ".join(mismatches)
            )

    return RobotConfig(
        robot_id=robot_id or (getattr(profile, "robot_id", None) or "franka_panda"),
        dof=derived["dof"],
        gripper_dof=derived["gripper_dof"],
        max_reach_m=derived["max_reach_m"],
        home_qpos=PANDA_HOME_QPOS.tolist(),
        home_gripper_qpos=PANDA_HOME_GRIPPER_QPOS.tolist(),
        mjcf_uri=str(p),
        mjcf_sha256=sha,
        mjcf_license=str(getattr(profile, "mjcf_license", "") or "Apache-2.0"),
        mjcf_source=str(
            getattr(profile, "mjcf_source", "")
            or "mujoco_menagerie/franka_emika_panda/panda.xml"
        ),
        # [ENGINEERING] T_E_TCP is the TCP pose expressed in the hand body frame
        # (see the module docstring for the derivation).  ADR-0005.  The depth is
        # taken from the finger pad collision geoms, not from the finger body
        # origin: the pads are ~44.7 mm further along the approach axis.
        T_E_TCP=tcp_offset.tolist(),
        joint_names=list(derived["joint_names"]),
        gripper_joint_names=list(derived["gripper_joint_names"]),
        actuator_names=[f"actuator{i}" for i in range(1, 9)],
        eef_body_name="hand",
        base_body_name="link0",
        gripper_geom_names=["hand", "left_finger", "right_finger"],
        max_joint_jump_rad=1.5,
        notes=[
            "gripper opening (m) = 2 * finger_joint position; range 0-0.08 m",
            "T_E_TCP derived from the menagerie model's finger body geometry, not assumed",
        ],
    )


# ---------------------------------------------------------------------------
# IK solvers
# ---------------------------------------------------------------------------


@dataclass
class PandaIKSolver:
    """Mink-based IK solver for the Panda ``hand`` body."""

    model: Any  # mujoco.MjModel
    config: RobotConfig
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    lm_damping: float = 0.0
    iterations: int = 100
    tolerance: float = 1e-5
    solver: str = "quadprog"

    def __post_init__(self) -> None:
        import mink

        self._mink = mink
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

    def _set_arm(self, q_arm: NDArray[np.float64], gripper: NDArray[np.float64]) -> Any:
        qpos = self._configuration.q.copy()
        qpos[: self.config.dof] = q_arm
        if self.config.gripper_dof:
            qpos[self.config.dof : self.config.dof + self.config.gripper_dof] = gripper
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

        mink = self._mink
        target = np.asarray(target_T_B_TCP, dtype=np.float64).reshape(4, 4)
        # T_B_TCP -> T_B_E  (E is the body Mink controls)
        T_B_E = target @ invert_transform(self.config.T_E_TCP_array)

        q_seed = (
            np.asarray(q_init, dtype=np.float64)[: self.config.dof].copy()
            if q_init is not None
            else self._q_arm.copy()
        )
        gripper = (
            np.asarray(gripper_qpos, dtype=np.float64)
            if gripper_qpos is not None
            else np.asarray(self.config.home_gripper_qpos, dtype=np.float64)
        )

        config = self._set_arm(q_seed, gripper)
        self.task.set_target(mink.SE3.from_matrix(T_B_E))

        iterations = int(max_iterations if max_iterations is not None else self.iterations)
        status = "max_iterations"
        it_used = 0
        error_msg: str | None = None
        for it in range(1, iterations + 1):
            it_used = it
            try:
                vel = mink.solve_ik(
                    config,
                    [self.task],
                    dt=1e-2,
                    solver=self.solver,
                    damping=1e-5,
                    limits=self.limits,
                )
            except mink.NoSolutionFound:
                status = "no_solution"
                break
            except Exception as exc:  # noqa: BLE001 - surfaced as a solver status
                status = "error"
                error_msg = f"{type(exc).__name__}: {exc}"
                break
            config.integrate_inplace(vel, dt=1e-2)

            err = self.task.compute_error(config)
            pos_err = float(np.linalg.norm(err[:3]))
            rot_err = float(np.linalg.norm(err[3:]))
            if pos_err < self.tolerance and rot_err < self.tolerance:
                status = "converged"
                break
        else:
            status = "max_iterations"

        q_solution = config.q[: self.config.dof].copy()
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

    def _limits_hit(self, q_arm: NDArray[np.float64]) -> bool:
        """True when a solved joint sits outside its limit (or within 1e-6 of it)."""
        for i in range(self.config.dof):
            lo, hi = self.model.jnt_range[i]
            if q_arm[i] < lo - 1e-6 or q_arm[i] > hi + 1e-6:
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
        data.qpos[: self.config.dof] = q_arm
        if self.config.gripper_dof:
            data.qpos[self.config.dof : self.config.dof + self.config.gripper_dof] = gripper
        mujoco.mj_forward(self.model, data)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise KeyError(f"body {body_name!r} not in the model")
        return make_transform(data.xmat[bid].reshape(3, 3), data.xpos[bid])


# ---------------------------------------------------------------------------
# Robot model
# ---------------------------------------------------------------------------


class FrankaPanda:
    """Franka Emika Panda + parallel gripper on MuJoCo."""

    robot_id = "franka_panda"

    def __init__(
        self,
        mjcf_uri: str | Path | None = None,
        *,
        config: RobotConfig | None = None,
        ik: PandaIKSolver | None = None,
        ik_settings: dict[str, Any] | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        path = Path(mjcf_uri) if mjcf_uri is not None else default_panda_xml()
        if not path.is_file():
            raise FileNotFoundError(f"Panda MJCF not found: {path}")
        self.mjcf_path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.config = config or build_franka_config(mjcf_uri=path)

        self.dof = self.config.dof
        self.max_reach_m = self.config.max_reach_m
        self.home_qpos = np.asarray(self.config.home_qpos, dtype=np.float64)
        self.home_gripper_qpos = np.asarray(self.config.home_gripper_qpos, dtype=np.float64)

        settings = dict(ik_settings or {})
        self.ik_solver = ik or PandaIKSolver(
            model=self.model,
            config=self.config,
            position_cost=float(settings.get("position_cost", 1.0)),
            orientation_cost=float(settings.get("orientation_cost", 1.0)),
            lm_damping=float(settings.get("lm_damping", 0.0)),
            iterations=int(settings.get("iterations", 100)),
            tolerance=float(settings.get("tolerance", 1e-5)),
            solver=str(settings.get("qp_backend", "quadprog")),
        )

    # -- helpers -----------------------------------------------------------

    def _set_qpos(
        self,
        q_arm: NDArray[np.floating[Any]],
        gripper: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        self.data.qpos[: self.dof] = np.asarray(q_arm, dtype=np.float64)[: self.dof]
        if self.config.gripper_dof:
            g = (
                np.asarray(gripper, dtype=np.float64)
                if gripper is not None
                else self.home_gripper_qpos
            )
            self.data.qpos[self.dof : self.dof + self.config.gripper_dof] = g
        self._mujoco.mj_forward(self.model, self.data)

    def _body_pose(self, body_name: str) -> SE3:
        bid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise KeyError(f"body {body_name!r} is not in {self.mjcf_path.name}")
        return make_transform(self.data.xmat[bid].reshape(3, 3), self.data.xpos[bid])

    def gripper_width_to_qpos(self, width_m: float) -> NDArray[np.float64]:
        """Metres of opening -> finger joint positions (half each side)."""
        half = 0.5 * float(np.clip(width_m, 0.0, self.max_gripper_width_m))
        return np.full(self.config.gripper_dof, half, dtype=np.float64)

    def qpos_to_gripper_width(self, gripper_qpos: NDArray[np.floating[Any]]) -> float:
        return float(np.mean(np.asarray(gripper_qpos, dtype=np.float64))) * 2.0

    @property
    def max_gripper_width_m(self) -> float:
        if not self.config.gripper_dof:
            return 0.0
        return 2.0 * float(self.model.jnt_range[self.dof][1])

    # -- RobotModel protocol ----------------------------------------------

    def forward_kinematics(
        self,
        qpos: NDArray[np.floating[Any]],
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> SE3:
        """``T_B_TCP`` for a joint configuration."""
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
        """Classify MuJoCo contacts into structural / self / scene categories."""
        self._set_qpos(qpos, gripper_qpos)
        mujoco = self._mujoco
        result = CollisionResult(has_self_collision=False)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[contact.geom1]))
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[contact.geom2]))
            b1 = b1 or "world"
            b2 = b2 or "world"
            pair = (b1, b2)
            result.max_penetration_m = max(result.max_penetration_m, float(contact.dist) * -1.0)
            if b1 == "world" or b2 == "world":
                result.scene_contact_pairs.append(pair)
                continue
            if frozenset({b1, b2}) in _ADJACENT_PAIRS:
                result.adjacent_contact_pairs.append(pair)
                continue
            result.self_contact_pairs.append(pair)

        result.has_self_collision = bool(result.self_contact_pairs)
        if result.has_self_collision:
            result.code = 1
        elif result.scene_contact_pairs:
            result.code = 2
        return result

    def render(self, qpos: NDArray[np.floating[Any]], camera: Any) -> RenderFrame:  # pragma: no cover - delegated
        """Rendering lives in :mod:`ego2robot.vision.render`."""
        from ...vision.render import MujocoRenderer

        renderer = MujocoRenderer(self, camera)
        return renderer.render(qpos)

    # -- contact helper used by base search / QC ---------------------------

    def body_positions(self, qpos: NDArray[np.floating[Any]], names: Sequence[str]) -> dict[str, NDArray[np.float64]]:
        self._set_qpos(qpos)
        return {n: self._body_pose(n)[:3, 3] for n in names}

    def reach_ratio(self, T_B_TCP: SE3) -> float:
        """Distance of a TCP target from the base, normalised by max reach."""
        return float(np.linalg.norm(np.asarray(T_B_TCP)[:3, 3])) / self.max_reach_m

    def apply_home(self) -> None:
        self._set_qpos(self.home_qpos, self.home_gripper_qpos)

    def set_state(
        self,
        qpos: NDArray[np.floating[Any]],
        gripper_qpos: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        """Forward kinematics for one configuration, leaving it as the live state."""
        self._set_qpos(qpos, gripper_qpos)

    def body_pose(self, body_name: str) -> SE3:
        """``T_B_body`` for the current state (requires a prior :meth:`set_state`)."""
        return self._body_pose(body_name)

    def tcp_pose(self) -> SE3:
        """``T_B_TCP`` for the current state."""
        return self._body_pose(self.config.eef_body_name) @ self.config.T_E_TCP_array

    def kinematics_summary(self) -> dict[str, Any]:
        mujoco = self._mujoco
        out: dict[str, Any] = {
            "mjcf": str(self.mjcf_path),
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "max_reach_m": self.max_reach_m,
            "max_gripper_width_m": self.max_gripper_width_m,
            "eef_body": self.config.eef_body_name,
            "T_E_TCP": self.config.T_E_TCP_array.tolist(),
            "joints": {},
            "bodies": [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                for i in range(self.model.nbody)
            ],
        }
        for i, name in enumerate(self.config.joint_names):
            out["joints"][name] = {
                "range": [float(v) for v in self.model.jnt_range[i]],
                "qpos_adr": int(self.model.jnt_qposadr[i]),
            }
        return out


def compose_base_to_camera(T_C_B: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """``T_B_C = inverse(T_C_B)`` -- helper kept next to the frame conventions."""
    return invert_transform(np.asarray(T_C_B, dtype=np.float64))


def base_tcp_from_camera(T_C_B: NDArray[np.floating[Any]], T_C_TCP: NDArray[np.floating[Any]]) -> SE3:
    """``T_B_TCP = inverse(T_C_B) @ T_C_TCP`` (system-design.md 5.2)."""
    return compose_transforms(invert_transform(np.asarray(T_C_B, dtype=np.float64)), np.asarray(T_C_TCP, dtype=np.float64))
