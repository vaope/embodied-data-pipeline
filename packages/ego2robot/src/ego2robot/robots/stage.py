"""``robot`` stage: canonical ``T_C_TCP`` -> selected robot ``qpos`` with FK validation.

This is milestone M3 and the project's most important gate:
``T_C_TCP -> T_B_TCP -> qpos -> FK -> image projection``.

Design decisions implemented literally:

* system-design.md 8.1 — ``q[t-1]`` seeds ``q[t]`` to suppress IK branch flips.
* system-design.md 8.3 — every frame records solver status, tracking error,
  joint-limit status, self-collision, adjacent joint jump and the FK-vs-target
  closure error.
* implementation-plan.md M3 — a failed frame gets an explicit status code; it is
  never silently overwritten by the previous frame's solution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core import artifacts as A
from ..core.capabilities import HandCapabilityError, require_sequence_support
from ..core.frames import Frame, transform_name
from ..core.schema import (
    CanonicalActionSequence,
    EpisodeManifest,
    RobotTrajectory,
    ValidationResult,
)
from ..core.transforms import geodesic_distance_rad, invert_transform, transform_points
from ..pipeline.registry import register_stage
from ..pipeline.stage import StageContext, StageResult
from .base_search import (
    BaseSearchConfig,
    load_fixed_base,
    search_base_placement,
)
from .ik import (
    COLLISION_NONE,
    COLLISION_SCENE,
    COLLISION_SELF,
    STATUS_INVALID_TARGET,
    STATUS_JOINT_LIMIT,
    STATUS_MAX_ITERATIONS,
    STATUS_NO_SOLUTION,
    STATUS_NOT_ATTEMPTED,
    STATUS_OK,
    STATUS_SOLVER_ERROR,
    STATUS_NAMES,
    solve_trajectory,
)

__all__ = ["RobotStage"]


def _episode(ctx: StageContext) -> EpisodeManifest:
    if not ctx.layout.episode_json.is_file():
        raise FileNotFoundError(f"{ctx.layout.episode_json} missing; run 'ingest' first")
    return EpisodeManifest.model_validate(A.read_json(ctx.layout.episode_json))


def _load_base(ctx: StageContext, episode: EpisodeManifest) -> tuple[np.ndarray, str, list[str]]:
    """Resolve ``T_C_B`` from the episode, then the profile.

    ``base.json`` written by a previous run of this stage is authoritative for
    reproducibility, so a cached placement is reused rather than recomputed.
    """
    base_path = ctx.layout.base_json
    if base_path.is_file():
        payload = A.read_json(base_path)
        if "T_C_B" in payload:
            return (
                np.asarray(payload["T_C_B"], dtype=np.float64),
                f"base.json ({payload.get('mode', 'unknown')})",
                list(payload.get("notes", [])),
            )
    profile_base = ctx.config.case.get("T_C_B")
    result = load_fixed_base(episode_T_C_B=episode.T_C_B, profile_T_C_B=profile_base)
    return result.T_C_B, result.source, result.notes


@register_stage
class RobotStage:
    """``robot``: base placement + per-frame IK/FK + contact QC."""

    name = "robot"
    version = "1"

    # -- contract ----------------------------------------------------------

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        res = ValidationResult()
        if not ctx.layout.actions_npz.is_file():
            res.add(
                "robot.no_canonical_actions",
                f"{ctx.layout.actions_npz} missing; run the 'retarget' stage first",
            )
            return res
        try:
            episode = _episode(ctx)
        except FileNotFoundError as exc:
            res.add("robot.no_episode", str(exc))
            return res
        has_base = (
            ctx.layout.base_json.is_file()
            or episode.T_C_B is not None
            or ctx.config.case.get("T_C_B") is not None
            or ctx.config.robot_binding.base_placement.mode == "grid_search"
        )
        if not has_base:
            res.add(
                "robot.no_base_placement",
                "no T_C_B in the episode, the pipeline profile, or a previous base.json; "
                "the MVP path requires an explicit fixed base (system-design.md 8.2)",
            )
        try:
            action = CanonicalActionSequence.load_npz(ctx.layout.actions_npz)
        except Exception as exc:  # noqa: BLE001 - reported as a validation issue
            res.add("robot.unreadable_actions", str(exc))
            return res
        res.extend(action.validate())
        try:
            require_sequence_support("robot", action)
        except HandCapabilityError as exc:
            res.add("robot.hand_count_unsupported", str(exc))
            return res
        if res.ok and not np.any(np.asarray(action["valid"])):
            res.add(
                "robot.no_valid_frames",
                "the canonical action sequence has no valid frames; IK would have nothing to solve",
            )
        return res

    def fingerprint(self, ctx: StageContext) -> str:
        from .factory import adapter_source, model_asset_hashes, resolve_mjcf

        retarget_manifest = A.read_stage_manifest(ctx.layout.manifest_path("retarget"))
        robot_profile = ctx.config.case.get("robot_profile", "")
        mjcf = resolve_mjcf(ctx)
        return A.hash_json(
            {
                "retarget_fingerprint": retarget_manifest.input_fingerprints.get("self", "")
                if retarget_manifest
                else "",
                "actions_sha": A.file_sha256(ctx.layout.actions_npz)
                if ctx.layout.actions_npz.is_file()
                else "",
                "robot_binding": ctx.config.robot_binding.model_dump(mode="json"),
                "model_assets": model_asset_hashes(mjcf),
                "adapter_code_sha": A.file_sha256(adapter_source(ctx.config.robot_binding.robot_id)),
                "robot_stage_code_sha": A.file_sha256(Path(__file__)),
                "robot_factory_code_sha": A.file_sha256(Path(__file__).with_name("factory.py")),
                "robot_profile": robot_profile,
                "base_json": A.file_sha256(ctx.layout.base_json)
                if ctx.layout.base_json.is_file()
                else "",
            }
        )

    def run(self, ctx: StageContext) -> StageResult:
        result = StageResult(stage=self.name)
        cfg = ctx.config.robot_binding
        log = ctx.log
        episode = _episode(ctx)

        from .factory import build_robot

        robot_config, robot = build_robot(
            ctx,
            ik_settings={
                "position_cost": cfg.ik.position_cost,
                "orientation_cost": cfg.ik.orientation_cost,
                "lm_damping": cfg.ik.lm_damping,
                "iterations": cfg.ik.iterations,
                "tolerance": cfg.ik.tolerance,
                "qp_backend": cfg.ik.qp_backend,
            },
        )
        result.metrics["robot_kinematics"] = {
            "mjcf": str(robot.mjcf_path),
            "mjcf_sha256": robot_config.mjcf_sha256[:16],
            "nq": int(robot.model.nq),
            "nu": int(robot.model.nu),
            "max_reach_m": robot.max_reach_m,
            "max_gripper_width_m": round(robot.max_gripper_width_m, 6),
            "T_E_TCP": robot_config.T_E_TCP,
            "ik_solver": f"mink/{cfg.ik.qp_backend}",
            "ik_iterations": cfg.ik.iterations,
            "ik_tolerance": cfg.ik.tolerance,
        }

        action = CanonicalActionSequence.load_npz(ctx.layout.actions_npz)
        # Defense in depth: validate_inputs already refuses H>1, but a direct
        # `run()` call must not be able to squeeze a two-hand sequence either.
        require_sequence_support("robot", action)
        T_C_TCP = np.asarray(action["T_C_TCP"], dtype=np.float64)[:, 0]
        valid = np.asarray(action["valid"], dtype=np.bool_)[:, 0]
        width = np.asarray(action["gripper_width_m"], dtype=np.float64)[:, 0]
        timestamps = np.asarray(action["timestamp_s"], dtype=np.float64)
        frame_index = np.asarray(action["frame_index"], dtype=np.int64)
        finite_width = valid & np.isfinite(width)
        below_width = finite_width & (width < cfg.gripper_width_min_m)
        above_width = finite_width & (width > cfg.gripper_width_max_m)
        result.metrics["gripper_binding"] = {
            "actuated": bool(robot_config.gripper_dof),
            "requested_width_min_m": float(np.min(width[finite_width])) if np.any(finite_width) else None,
            "requested_width_max_m": float(np.max(width[finite_width])) if np.any(finite_width) else None,
            "model_width_min_m": float(cfg.gripper_width_min_m) if robot_config.gripper_dof else None,
            "model_width_max_m": float(cfg.gripper_width_max_m) if robot_config.gripper_dof else None,
            "below_range_frames": int(np.count_nonzero(below_width)) if robot_config.gripper_dof else None,
            "above_range_frames": int(np.count_nonzero(above_width)) if robot_config.gripper_dof else None,
            "unsupported_width_frames": int(np.count_nonzero(finite_width)) if not robot_config.gripper_dof else 0,
        }
        if robot_config.gripper_dof and np.any(below_width | above_width):
            result.warnings.append(
                f"{np.count_nonzero(below_width | above_width)} gripper widths are outside "
                "the model range and will be clamped by the robot adapter"
            )
        if not robot_config.gripper_dof:
            result.warnings.append("selected robot has no gripper actuation; TCP tracking is pose-only")

        # ---- base placement ------------------------------------------------
        base_mode = cfg.base_placement.mode
        base_notes: list[str] = []
        if base_mode == "grid_search" and not ctx.layout.base_json.is_file():
            search_cfg = BaseSearchConfig(
                lateral_levels=tuple(cfg.base_placement.lateral_levels),
                forward_levels=tuple(cfg.base_placement.forward_levels),
                vertical_levels=tuple(cfg.base_placement.vertical_levels),
                pitch_deg=tuple(cfg.base_placement.pitch_deg),
                yaw_deg=tuple(cfg.base_placement.yaw_deg),
                roll_deg=tuple(cfg.base_placement.roll_deg),
                min_camera_distance_m=cfg.base_placement.min_camera_distance_m,
                min_base_distance_m=cfg.base_placement.min_base_distance_m,
                max_reach_fraction=cfg.base_placement.max_reach_fraction,
                max_keyframes=cfg.base_placement.max_keyframes,
                reach_ratio_target=cfg.base_placement.reach_ratio_target,
                reach_penalty_weight=cfg.base_placement.reach_penalty_weight,
                ik_iterations=cfg.base_placement.ik_iterations,
                ik_tolerance=cfg.base_placement.ik_tolerance,
            )
            search = search_base_placement(robot, T_C_TCP, valid, cfg=search_cfg, log=log)
            T_C_B = search.T_C_B
            base_payload = search.to_json(robot_id=robot_config.robot_id)
            base_source = "grid_search"
            full_ok, full_ratio = _validate_full_trajectory(robot, T_C_B, T_C_TCP, valid, width, cfg)
            base_payload["full_trajectory_validation"] = {
                "ik_success_ratio": round(full_ratio, 6),
                "meets_min_ratio": bool(full_ok),
            }
            if not full_ok:
                result.warnings.append(
                    f"grid-search winner only achieves IK success {full_ratio:.3f} over the full "
                    "trajectory; the MVP fixed-base gate is not met"
                )
            if log:
                log.info(
                    f"base search selected candidate {search.best.index} "
                    f"(FR={search.best.feasibility_rate:.3f}, "
                    f"reach={search.best.mean_reach_ratio:.3f}); full-trajectory IK {full_ratio:.3f}"
                )
        else:
            T_C_B, base_source, base_notes = _load_base(ctx, episode)
            if ctx.layout.base_json.is_file():
                base_payload = A.read_json(ctx.layout.base_json)
                if base_payload.get("robot_id") != robot_config.robot_id:
                    raise ValueError("cached base.json belongs to a different robot morphology")
            else:
                base_payload = {
                    "schema_version": "ego2robot.base.v1",
                    "mode": "fixed",
                    "pose_variable": str(transform_name(Frame.C, Frame.B)),
                    "T_C_B": T_C_B.tolist(),
                    "source": base_source,
                    "notes": base_notes,
                    "robot_id": robot_config.robot_id,
                }
        result.add_output(A.atomic_write_json(ctx.layout.base_json, base_payload))
        result.metrics["base"] = {
            "mode": base_payload["mode"],
            "source": base_source,
            "T_C_B": np.round(T_C_B, 6).tolist(),
            "notes": base_notes,
        }

        # ---- per-frame IK --------------------------------------------------
        # The loop lives in ``robots/ik.py`` so the bimanual rig stage solves each
        # arm with exactly the same warm start / seed recovery / status codes.
        T = int(T_C_TCP.shape[0])
        arm = solve_trajectory(
            robot=robot,
            robot_config=robot_config,
            ik=cfg.ik,
            targets_B=np.einsum("ij,tjk->tik", invert_transform(T_C_B), T_C_TCP),
            valid=valid,
            width_m=width,
            timestamps=timestamps,
            gripper_width_min_m=float(cfg.gripper_width_min_m),
        )
        qpos = arm.qpos
        qvel = arm.qvel
        gripper_qpos = arm.gripper_qpos
        T_B_TCP_target = arm.T_B_TCP_target
        T_B_TCP_fk = arm.T_B_TCP_fk
        pos_err = arm.position_error_m
        rot_err = arm.rotation_error_rad
        ik_success = arm.ik_success
        collision_code = arm.collision_code
        joint_limit_hit = arm.joint_limit_hit
        ik_iterations = arm.ik_iterations
        solver_status = arm.solver_status
        reach_ratios = arm.reach_ratios
        branch_jumps = arm.branch_jumps

        trajectory = RobotTrajectory(
            {
                "timestamp_s": timestamps,
                "frame_index": frame_index,
                "qpos": qpos,
                "qvel": qvel,
                "gripper_qpos": gripper_qpos,
                "T_B_TCP_target": T_B_TCP_target[:, None, :, :],
                "T_B_TCP_fk": T_B_TCP_fk[:, None, :, :],
                "position_error_m": pos_err[:, None],
                "rotation_error_rad": rot_err[:, None],
                "ik_success": ik_success[:, None],
                "collision_code": collision_code,
                "joint_limit_hit": joint_limit_hit,
                "ik_iterations": ik_iterations,
                "solver_status": solver_status,
            },
            meta={
                "pose_variable": str(transform_name(Frame.B, Frame.TCP)),
                "robot_id": robot_config.robot_id,
                "T_C_B": T_C_B.tolist(),
                "T_E_TCP": robot_config.T_E_TCP,
                "mjcf": str(robot.mjcf_path),
                "mjcf_sha256": robot_config.mjcf_sha256,
                "ik_solver": f"mink/{cfg.ik.qp_backend}",
                "status_registry": {str(k): v for k, v in STATUS_NAMES.items()},
                "collision_registry": {"0": "none", "1": "self", "2": "scene"},
                "gripper_binding": result.metrics["gripper_binding"],
                "gripper_width_to_qpos": next(
                    (note for note in robot_config.notes if "gripper" in note.lower()),
                    "see robot profile and adapter; no gripper actuation if gripper_dof=0",
                ),
            },
        )
        traj_check = trajectory.validate()
        traj_check.raise_if_bad()
        result.add_output(trajectory.save_npz(ctx.layout.trajectory_npz))

        # ---- metrics -------------------------------------------------------
        n_valid = int(np.count_nonzero(valid))
        solved = int(np.count_nonzero(ik_success))
        finite_mask = np.isfinite(pos_err) & np.isfinite(rot_err)
        result.metrics.update(
            {
                "num_frames": T,
                "ik_tolerance_m": float(cfg.ik.tolerance),
                "valid_frames": n_valid,
                "ik_attempted": n_valid,
                "ik_success": solved,
                "ik_success_ratio": round(solved / n_valid, 6) if n_valid else 0.0,
                "ik_success_ratio_over_all_frames": round(solved / T, 6) if T else 0.0,
                "p95_position_error_m": round(float(np.percentile(pos_err[finite_mask], 95)), 9)
                if finite_mask.any()
                else None,
                "p95_position_error_over_all_frames_m": round(float(np.nanpercentile(pos_err, 95)), 9)
                if T
                else None,
                "max_position_error_m": round(float(np.nanmax(pos_err)), 9) if T else None,
                "p95_rotation_error_rad": round(float(np.percentile(rot_err[finite_mask], 95)), 9)
                if finite_mask.any()
                else None,
                "p95_rotation_error_over_all_frames_rad": round(float(np.nanpercentile(rot_err, 95)), 9)
                if T
                else None,
                "max_rotation_error_rad": round(float(np.nanmax(rot_err)), 9) if T else None,
                "self_collision_frames": int(np.count_nonzero(collision_code == COLLISION_SELF)),
                "scene_contact_frames": int(np.count_nonzero(collision_code == COLLISION_SCENE)),
                "joint_limit_frames": int(np.count_nonzero(joint_limit_hit)),
                "mean_reach_ratio": round(float(np.mean(reach_ratios)), 6) if reach_ratios else None,
                "max_reach_ratio": round(float(np.max(reach_ratios)), 6) if reach_ratios else None,
                "max_joint_jump_rad": round(float(np.max(branch_jumps)), 6) if branch_jumps else 0.0,
                "joint_jump_frames": int(np.count_nonzero(np.asarray(branch_jumps) > cfg.ik.max_joint_jump_rad))
                if branch_jumps
                else 0,
                "solver_status_histogram": {
                    STATUS_NAMES.get(int(s), str(s)): int(np.count_nonzero(solver_status == s))
                    for s in np.unique(solver_status)
                },
                "ik_iterations_mean": round(float(np.mean(ik_iterations[np.asarray(ik_iterations) > 0])), 3)
                if np.any(np.asarray(ik_iterations) > 0)
                else 0.0,
            }
        )

        result.payload["trajectory"] = trajectory
        result.payload["T_C_B"] = T_C_B
        result.payload["robot"] = robot
        result.payload["robot_config"] = robot_config
        result.payload["action"] = action
        return result

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        res = ValidationResult()
        traj = result.payload.get("trajectory")
        if not isinstance(traj, RobotTrajectory):
            res.add("robot.no_trajectory", "stage produced no RobotTrajectory payload")
            return res
        res.extend(traj.validate())
        res.extend(
            traj.validate_numeric(
                allow_nan_in=("T_B_TCP_target", "T_B_TCP_fk", "position_error_m", "rotation_error_rad")
            )
        )
        if res.ok:
            success = np.asarray(traj["ik_success"])[:, 0]
            status = np.asarray(traj["solver_status"])
            target = np.asarray(traj["T_B_TCP_target"])[:, 0]
            fk = np.asarray(traj["T_B_TCP_fk"])[:, 0]
            # the closure check re-derives T_B_TCP from the stored qpos and must
            # reproduce the target within a small multiple of the solver's own
            # tolerance (the paper's 1e-5 threshold, appendix A.4).  The solver
            # stops on its *task error*, so re-deriving the pose can land a few
            # percent outside that number; a factor of 5 is still two orders of
            # magnitude below the MVP geometry gate.
            closure_tol = float(result.metrics.get("ik_tolerance_m", 1e-5)) * 5.0
            for t in np.nonzero(success)[0]:
                closure = np.linalg.norm(fk[t][:3, 3] - target[t][:3, 3])
                if closure > closure_tol:
                    res.add(
                        "robot.fk_closure",
                        f"frame {t}: FK-vs-target closure error {closure:.3e} m exceeds the "
                        f"solver tolerance {closure_tol:.1e} m",
                    )
                    break
            if not np.any(success):
                res.add(
                    "robot.zero_ik_success",
                    "no frame solved IK; refusing to publish a trajectory that looks complete",
                )
            unknown = set(np.unique(status).tolist()) - set(STATUS_NAMES)
            if unknown:
                res.add("robot.unknown_status", f"unknown solver status code(s): {sorted(unknown)}")
        for path in result.outputs:
            if not Path(path).is_file():
                res.add("robot.missing_output", f"declared output does not exist: {path}")
        return res


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validate_full_trajectory(
    robot: Any,
    T_C_B: np.ndarray,
    T_C_TCP: np.ndarray,
    valid: np.ndarray,
    width: np.ndarray,
    cfg: Any,
) -> tuple[bool, float]:
    """Run IK over the whole trajectory to confirm a keyframe-ranked candidate."""
    T_B_C = invert_transform(T_C_B)
    q_prev = np.asarray(robot.config.home_qpos, dtype=np.float64)
    solved = 0
    attempted = 0
    for t in np.nonzero(valid)[0]:
        target = T_B_C @ T_C_TCP[t]
        if not np.isfinite(target).all():
            continue
        attempted += 1
        res = robot.solve_ik(
            target, q_prev, gripper_qpos=robot.gripper_width_to_qpos(float(width[t]))
        )
        if res.success:
            solved += 1
            q_prev = res.qpos
    ratio = solved / attempted if attempted else 0.0
    from ..core.config import QualityConfig

    minimum = cfg.mvp_min_ik_success_ratio if isinstance(cfg, QualityConfig) else 0.95
    return bool(ratio >= minimum), ratio
