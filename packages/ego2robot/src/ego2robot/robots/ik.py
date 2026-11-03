"""Per-arm IK/FK trajectory solve, shared by the single-arm and rig stages.

The frame loop is subtle enough (warm start, deterministic branch recovery,
status codes, collision classification, reach/branch diagnostics) that having two
copies would guarantee they drift.  It therefore lives here, and both the
single-arm ``robot`` stage and the bimanual ``rig`` stage call
:func:`solve_trajectory`.

Behavior is frozen by evidence: after the bimanual migration the single-arm stage
was re-run on the four reviewed episodes and its ``trajectory.npz`` /
``actions*.npz`` are byte-identical to the pre-change copies
(``experiments/bimanual_b0b1_20261102/comparison.json``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ArmTrajectoryResult",
    "COLLISION_NONE",
    "COLLISION_SCENE",
    "COLLISION_SELF",
    "STATUS_FRAME_INVALID",
    "STATUS_INVALID_TARGET",
    "STATUS_JOINT_LIMIT",
    "STATUS_MAX_ITERATIONS",
    "STATUS_NO_SOLUTION",
    "STATUS_NOT_ATTEMPTED",
    "STATUS_OK",
    "STATUS_SKIPPED_UNREACHABLE",
    "STATUS_SOLVER_ERROR",
    "STATUS_NAMES",
    "joint_bounds",
    "seed_variant",
    "solve_trajectory",
]

# collision_code registry (uint16)
COLLISION_NONE = 0
COLLISION_SELF = 1
COLLISION_SCENE = 2

# solver_status registry (uint16)
STATUS_OK = 0
STATUS_MAX_ITERATIONS = 1
STATUS_NO_SOLUTION = 2
STATUS_SOLVER_ERROR = 3
STATUS_INVALID_TARGET = 4
STATUS_FRAME_INVALID = 5
STATUS_JOINT_LIMIT = 6
STATUS_NOT_ATTEMPTED = 7
STATUS_SKIPPED_UNREACHABLE = 8

STATUS_NAMES: dict[int, str] = {
    STATUS_OK: "ok",
    STATUS_MAX_ITERATIONS: "max_iterations",
    STATUS_NO_SOLUTION: "no_solution",
    STATUS_SOLVER_ERROR: "solver_error",
    STATUS_INVALID_TARGET: "invalid_target",
    STATUS_FRAME_INVALID: "frame_invalid",
    STATUS_JOINT_LIMIT: "joint_limit",
    STATUS_NOT_ATTEMPTED: "not_attempted",
    STATUS_SKIPPED_UNREACHABLE: "skipped_unreachable",
}


@dataclass
class ArmTrajectoryResult:
    """One arm's solved trajectory plus the diagnostics a QC report needs."""

    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    gripper_qpos: NDArray[np.float64]
    T_B_TCP_target: NDArray[np.float64]
    T_B_TCP_fk: NDArray[np.float64]
    position_error_m: NDArray[np.float64]
    rotation_error_rad: NDArray[np.float64]
    ik_success: NDArray[np.bool_]
    collision_code: NDArray[np.uint16]
    joint_limit_hit: NDArray[np.bool_]
    ik_iterations: NDArray[np.int32]
    solver_status: NDArray[np.uint16]
    reach_ratios: list[float] = field(default_factory=list)
    branch_jumps: list[float] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])

    def metrics(self, *, max_joint_jump_rad: float) -> dict[str, Any]:
        """Aggregate numbers, using the same definitions as the single-arm stage."""
        solved = int(np.count_nonzero(self.ik_success))
        finite_mask = np.isfinite(self.position_error_m) & np.isfinite(self.rotation_error_rad)
        jumps = np.asarray(self.branch_jumps, dtype=np.float64)
        return {
            "ik_success": solved,
            "p95_position_error_m": round(float(np.percentile(self.position_error_m[finite_mask], 95)), 9)
            if finite_mask.any()
            else None,
            "p95_rotation_error_rad": round(float(np.percentile(self.rotation_error_rad[finite_mask], 95)), 9)
            if finite_mask.any()
            else None,
            "max_position_error_m": round(float(np.nanmax(self.position_error_m)), 9)
            if self.num_frames
            else None,
            "max_rotation_error_rad": round(float(np.nanmax(self.rotation_error_rad)), 9)
            if self.num_frames
            else None,
            "self_collision_frames": int(np.count_nonzero(self.collision_code == COLLISION_SELF)),
            "scene_contact_frames": int(np.count_nonzero(self.collision_code == COLLISION_SCENE)),
            "joint_limit_frames": int(np.count_nonzero(self.joint_limit_hit)),
            "mean_reach_ratio": round(float(np.mean(self.reach_ratios)), 6) if self.reach_ratios else None,
            "max_reach_ratio": round(float(np.max(self.reach_ratios)), 6) if self.reach_ratios else None,
            "max_joint_jump_rad": round(float(np.max(jumps)), 6) if jumps.size else 0.0,
            "joint_jump_frames": int(np.count_nonzero(jumps > max_joint_jump_rad)) if jumps.size else 0,
            "solver_status_histogram": {
                STATUS_NAMES.get(int(s), str(s)): int(np.count_nonzero(self.solver_status == s))
                for s in np.unique(self.solver_status)
            },
            "ik_iterations_mean": round(float(np.mean(self.ik_iterations[self.ik_iterations > 0])), 3)
            if np.any(self.ik_iterations > 0)
            else 0.0,
        }


def joint_bounds(robot: Any, joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read arm-joint ranges from the selected MJCF, in trajectory order."""

    mujoco = robot._mujoco
    model = robot.model
    ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)) for name in joint_names]
    if any(joint_id < 0 for joint_id in ids):
        raise ValueError("configured arm joint is absent from the selected MJCF")
    if any(not model.jnt_limited[joint_id] for joint_id in ids):
        raise ValueError("deterministic IK seed generation requires finite MJCF arm-joint limits")
    bounds = np.asarray(model.jnt_range[ids], dtype=np.float64)
    return bounds[:, 0], bounds[:, 1]


def seed_variant(home: np.ndarray, lo: np.ndarray, hi: np.ndarray, attempt: int) -> np.ndarray:
    """Deterministic alternate IK seeds to escape an unreachable branch.

    Kept deterministic (a fixed RNG stream keyed by the attempt index) so a rerun
    with the same inputs solves the same frames -- resume and cache correctness
    depend on it.  The variants are:

    ``attempt % 4 == 0``  a uniform sample of the joint box
    ``attempt % 4 == 1``  the home pose perturbed with moderate noise
    ``attempt % 4 == 2``  the joint-space mirror of home around each joint's
                          own range midpoint, giving another IK branch
    ``attempt % 4 == 3``  a small perturbation of home, which resolves the common
                          case of a locally-blocked warm start
    """
    rng = np.random.default_rng(1000 + attempt)
    if home.shape != lo.shape or home.shape != hi.shape:
        raise ValueError("IK seed bounds must match the selected robot DOF")
    span = hi - lo
    dof = home.size
    kind = attempt % 4
    if kind == 0:
        seed = lo + rng.random(dof) * span
    elif kind == 1:
        seed = home + rng.normal(0.0, 0.6, dof)
    elif kind == 2:
        # flipping the sign of every joint is not a physical pose, so mirror the
        # four joints whose limits are symmetric about zero and re-centre the rest
        mid = 0.5 * (lo + hi)
        mirrored = mid + (mid - home)
        seed = mirrored
    else:
        seed = home + rng.normal(0.0, 0.25, dof)
    return np.clip(seed, lo, hi)


def solve_trajectory(
    *,
    robot: Any,
    robot_config: Any,
    ik: Any,
    targets_B: NDArray[np.float64],
    valid: NDArray[np.bool_],
    width_m: NDArray[np.float64],
    timestamps: NDArray[np.float64],
    gripper_width_min_m: float,
) -> ArmTrajectoryResult:
    """Solve every frame of one arm.

    ``targets_B`` is ``[T, 4, 4]`` in that arm's own base frame, ``valid``/``width_m``
    are ``[T]``.  A frame that is invalid, has a non-finite target or fails IK keeps
    its previous joint vector **and a non-zero status code**; it is never presented
    as a solved frame.
    """
    T = int(np.asarray(targets_B).shape[0])
    dof = int(robot.dof)
    gdof = int(getattr(robot_config, "gripper_dof", 0))

    qpos = np.zeros((T, dof), dtype=np.float64)
    qvel = np.zeros((T, dof), dtype=np.float64)
    gripper_qpos = np.zeros((T, gdof), dtype=np.float64)
    T_B_TCP_target = np.full((T, 4, 4), np.nan, dtype=np.float64)
    T_B_TCP_fk = np.full((T, 4, 4), np.nan, dtype=np.float64)
    pos_err = np.full(T, np.nan, dtype=np.float64)
    rot_err = np.full(T, np.nan, dtype=np.float64)
    ik_success = np.zeros(T, dtype=bool)
    collision_code = np.zeros(T, dtype=np.uint16)
    joint_limit_hit = np.zeros(T, dtype=bool)
    ik_iterations = np.zeros(T, dtype=np.int32)
    solver_status = np.full(T, STATUS_NOT_ATTEMPTED, dtype=np.uint16)

    valid = np.asarray(valid, dtype=bool)
    q_prev = np.asarray(robot_config.home_qpos, dtype=np.float64)
    g_prev = robot.gripper_width_to_qpos(float(gripper_width_min_m))
    reach_ratios: list[float] = []
    branch_jumps: list[float] = []
    seed_attempts = int(ik.seed_attempts) if ik.seed_attempts else 1
    joint_lower, joint_upper = joint_bounds(robot, list(robot_config.joint_names))

    for t in range(T):
        target = np.asarray(targets_B[t], dtype=np.float64)
        T_B_TCP_target[t] = target
        target_finite = bool(np.isfinite(target).all())
        gripper_cmd = (
            robot.gripper_width_to_qpos(float(width_m[t])) if np.isfinite(width_m[t]) else g_prev
        )

        if not valid[t] or not target_finite:
            solver_status[t] = STATUS_FRAME_INVALID if not valid[t] else STATUS_INVALID_TARGET
            qpos[t] = q_prev
            gripper_qpos[t] = gripper_cmd
            continue

        res = robot.solve_ik(
            target,
            q_prev if ik.warm_start else None,
            gripper_qpos=gripper_cmd,
        )
        if not res.success and seed_attempts > 1:
            # Branch recovery with deterministic seeds.  ``home`` is tried first
            # because it is the one configuration known to be collision-free and
            # well inside the joint box; the rest are the numbered variants.
            best = res
            home_seed = np.asarray(robot_config.home_qpos, dtype=np.float64)
            tried = [home_seed] + [
                seed_variant(home_seed, joint_lower, joint_upper, a)
                for a in range(max(0, seed_attempts - 1))
            ]
            for seed in tried:
                if np.allclose(seed, q_prev):
                    continue  # already failed with this seed
                cand = robot.solve_ik(target, seed, gripper_qpos=gripper_cmd)
                if cand.success:
                    best = cand
                    break
                if cand.position_error_m < best.position_error_m:
                    best = cand
            res = best

        T_B_TCP_fk[t] = res.T_B_TCP_fk
        pos_err[t] = res.position_error_m
        rot_err[t] = res.rotation_error_rad
        ik_success[t] = res.success
        ik_iterations[t] = res.iterations
        joint_limit_hit[t] = res.joint_limit_hit
        if res.status == "converged":
            solver_status[t] = STATUS_JOINT_LIMIT if res.joint_limit_hit else STATUS_OK
        else:
            solver_status[t] = {
                "max_iterations": STATUS_MAX_ITERATIONS,
                "no_solution": STATUS_NO_SOLUTION,
                "error": STATUS_SOLVER_ERROR,
            }.get(res.status, STATUS_SOLVER_ERROR)

        if res.success:
            qpos[t] = res.qpos
            gripper_qpos[t] = gripper_cmd
        else:
            # Never silently reuse the previous solution as if it had worked:
            # the status code carries the failure, and qpos repeats q_prev only
            # so the arrays stay writeable.
            qpos[t] = q_prev
            gripper_qpos[t] = gripper_cmd

        coll = robot.check_collision(qpos[t], gripper_qpos[t])
        collision_code[t] = (
            COLLISION_SELF
            if coll.has_self_collision
            else COLLISION_SCENE
            if coll.scene_contact_pairs
            else COLLISION_NONE
        )
        reach_ratios.append(float(np.linalg.norm(target[:3, 3])) / robot.max_reach_m)
        if t:
            branch_jumps.append(float(np.max(np.abs(qpos[t] - qpos[t - 1]))))
        q_prev = qpos[t]
        g_prev = gripper_qpos[t]

    for t in range(1, T):
        qvel[t] = (qpos[t] - qpos[t - 1]) / max(timestamps[t] - timestamps[t - 1], 1e-9)
    qvel[0] = qvel[1] if T > 1 else 0.0

    return ArmTrajectoryResult(
        qpos=qpos,
        qvel=qvel,
        gripper_qpos=gripper_qpos,
        T_B_TCP_target=T_B_TCP_target,
        T_B_TCP_fk=T_B_TCP_fk,
        position_error_m=pos_err,
        rotation_error_rad=rot_err,
        ik_success=ik_success,
        collision_code=collision_code,
        joint_limit_hit=joint_limit_hit,
        ik_iterations=ik_iterations,
        solver_status=solver_status,
        reach_ratios=reach_ratios,
        branch_jumps=branch_jumps,
    )
