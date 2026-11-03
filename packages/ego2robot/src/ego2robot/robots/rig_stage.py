"""``rig`` stage: two-arm IK/FK plus a joint collision scene (ADR-0022, B2).

What this stage is
------------------
PLAN-02 B2 has two halves.  This stage implements the first one: **two explicit
fixed bases, per-arm IK/FK, one MuJoCo scene for cross-arm contacts, and the
projection/3D evidence**.  The automatic pairwise two-base search is deliberately
left out (ADR-0022 decision 5) so the assembly and the search algorithm are not
validated at the same time.

What it refuses to do
---------------------
* It does not squeeze a two-hand canonical sequence to one arm: hand ids must cover
  ``right`` and ``left``, and each arm is bound to exactly one hand side.
* It does not reuse a single-arm artifact.  Every arm gets its own
  ``RobotTrajectory`` in that arm's base frame.
* It does not claim a visual result.  Cross-arm contacts and FK agreement are
  geometry; rendering, human removal and compositing are B3 and stay unimplemented.

Evidence produced (AGENTS.md rule 8)
------------------------------------
``robot/rig/<arm>/trajectory.npz``
    per-arm target/FK/errors/status/collision, exactly the single-arm schema.
``robot/rig/rig.json``
    binding, both bases, relative placement, generated scene path + sha, model and
    code hashes, per-arm and joint metrics, explicit gate list.
``robot/rig/contacts.npz``
    per-frame cross-arm contact counts, deduplicated pair counts, FK disagreement
    and the per-arm visibility proxy.
``previews/bimanual-*.png`` / ``.mp4``
    both arms projected onto the real frames plus a 3-D scene view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core import artifacts as A
from ..core.capabilities import require_sequence_support
from ..core.frames import Frame, transform_name
from ..core.schema import (
    CanonicalActionSequence,
    EpisodeManifest,
    RobotTrajectory,
    ValidationResult,
)
from ..core.transforms import invert_transform, make_transform
from ..pipeline.registry import register_stage
from ..pipeline.stage import StageContext, StageResult
from .base_search import BaseSearchConfig
from .factory import build_robot_for
from .ik import solve_trajectory
from .joint_search import select_joint_bases
from .rig import ARM_PREFIX, DualRigScene, build_dual_scene, relative_placement

__all__ = ["RigStage"]

#: Arm colour in the 2-D/3-D evidence (RGB).
ARM_COLOUR = {"right": (60, 120, 255), "left": (255, 150, 40)}
#: TCP target axes, kept the same semantic colours as the single-arm projection.
AXIS_RGB = ((255, 60, 60), (60, 220, 60), (60, 120, 255))
#: chain bodies drawn for one arm, in order (the Panda model's own names)
CHAIN = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand")


def _episode(ctx: StageContext) -> EpisodeManifest:
    if not ctx.layout.episode_json.is_file():
        raise FileNotFoundError(f"{ctx.layout.episode_json} missing; run 'ingest' first")
    return EpisodeManifest.model_validate(A.read_json(ctx.layout.episode_json))


def _rotation_error_rad(T_a: np.ndarray, T_b: np.ndarray) -> float:
    from ..core.transforms import geodesic_distance_rad

    return float(geodesic_distance_rad(T_a[:3, :3], T_b[:3, :3]))


def _joint_keyframes(action: CanonicalActionSequence, *, max_frames: int = 24) -> list[int]:
    """Keyframes for the joint base search (PLAN-02 B2).

    The union of: each hand's TCP extremes along x/y/z, the frame where the two TCPs
    are closest and farthest apart, and the frames of minimum/maximum relative
    rotation.  They are evenly thinned to ``max_frames`` so the pair evaluation stays
    bounded while still covering "two hands far apart", "two hands together" and "the
    two TCP orientations diverge".
    """
    pos = np.asarray(action["tcp_position_C_m"], dtype=np.float64)
    R = np.asarray(action["T_C_TCP"], dtype=np.float64)[:, :, :3, :3]
    T = pos.shape[0]
    if T == 0:
        return []
    picks: set[int] = {0, T - 1}
    for h in range(pos.shape[1]):
        for axis in range(3):
            column = pos[:, h, axis]
            if np.isfinite(column).any():
                picks.add(int(np.nanargmin(column)))
                picks.add(int(np.nanargmax(column)))
    if pos.shape[1] == 2:
        dist = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
        if np.isfinite(dist).any():
            picks.add(int(np.nanargmin(dist)))
            picks.add(int(np.nanargmax(dist)))
        angle = np.array(
            [
                _rotation_error_rad(R[t, 0], R[t, 1])
                if np.isfinite(R[t, 0]).all() and np.isfinite(R[t, 1]).all()
                else np.nan
                for t in range(T)
            ]
        )
        if np.isfinite(angle).any():
            picks.add(int(np.nanargmin(angle)))
            picks.add(int(np.nanargmax(angle)))
    ordered = sorted(picks)
    if len(ordered) <= max_frames:
        return ordered
    idx = np.linspace(0, len(ordered) - 1, max_frames).astype(int)
    return sorted({ordered[i] for i in idx})



def _owner_rgb(owner: np.ndarray) -> np.ndarray:
    """Colourise the ownership map: background black, right blue, left orange."""
    rgb = np.zeros(owner.shape + (3,), dtype=np.uint8)
    rgb[owner == 1] = (60, 120, 255)
    rgb[owner == 2] = (255, 150, 40)
    return rgb


@register_stage
class RigStage:
    """``rig``: two-arm geometry, cross-arm contacts, projection evidence."""

    name = "rig"
    version = "1"

    # -- contract ----------------------------------------------------------

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        res = ValidationResult()
        cfg = ctx.config.rig
        if not cfg.enabled:
            res.add("rig.disabled", "rig.enabled is false; this profile does not run a two-arm rig")
            return res
        if not ctx.layout.actions_npz.is_file():
            res.add(
                "rig.no_canonical_actions",
                f"{ctx.layout.actions_npz} missing; run the 'retarget' stage first",
            )
            return res
        try:
            action = CanonicalActionSequence.load_npz(ctx.layout.actions_npz)
        except Exception as exc:  # noqa: BLE001 - reported as a validation issue
            res.add("rig.unreadable_actions", str(exc))
            return res
        try:
            require_sequence_support("rig", action)
        except Exception as exc:  # noqa: BLE001 - a capability mismatch is a validation issue
            res.add("rig.hand_count_unsupported", str(exc))
            return res
        hand_ids = action.hand_ids
        if sorted(hand_ids) != ["left", "right"]:
            res.add(
                "rig.hand_ids_incomplete",
                f"the rig binds both hands, but the canonical action declares {hand_ids}",
            )
        for arm in cfg.arms:
            if arm.side not in hand_ids:
                res.add(
                    "rig.arm_side_missing",
                    f"arm {arm.arm_id!r} is bound to hand {arm.side!r}, which is absent "
                    f"from {hand_ids}",
                )
        try:
            _episode(ctx)
        except FileNotFoundError as exc:
            res.add("rig.no_episode", str(exc))
        if cfg.base_mode == "fixed":
            missing = [a.arm_id for a in cfg.arms if a.T_C_B is None]
            if missing:
                res.add("rig.fixed_base_missing", f"rig.base_mode is 'fixed' but arms {missing} have no T_C_B")
        return res

    def fingerprint(self, ctx: StageContext) -> str:
        from .factory import adapter_source, model_asset_hashes

        cfg = ctx.config.rig
        arms: dict[str, Any] = {}
        for arm in cfg.arms:
            try:
                robot_config, robot = build_robot_for(
                    arm.robot_id, repo_root=ctx.repo_root, mjcf_uri=ctx.config.case.get("mjcf_uri")
                )
                arms[arm.arm_id] = {
                    "robot_id": arm.robot_id,
                    "side": arm.side,
                    "mjcf_sha256": robot_config.mjcf_sha256,
                    "model_assets": model_asset_hashes(robot.mjcf_path),
                    "adapter_code_sha": A.file_sha256(adapter_source(arm.robot_id)),
                }
            except Exception as exc:  # noqa: BLE001 - a broken arm must surface, not be skipped
                arms[arm.arm_id] = {"error": f"{type(exc).__name__}: {exc}"}
        return A.hash_json(
            {
                "actions": A.file_sha256(ctx.layout.actions_npz)
                if ctx.layout.actions_npz.is_file()
                else "",
                "rig": cfg.model_dump(mode="json"),
                "ik": ctx.config.robot_binding.ik.model_dump(mode="json"),
                "arms": arms,
                "rig_stage_code_sha": A.file_sha256(Path(__file__)),
                "ik_code_sha": A.file_sha256(Path(__file__).with_name("ik.py")),
                "rig_code_sha": A.file_sha256(Path(__file__).with_name("rig.py")),
                "stage_version": self.version,
            }
        )

    # -- run ---------------------------------------------------------------

    def run(self, ctx: StageContext) -> StageResult:
        result = StageResult(stage=self.name)
        cfg = ctx.config.rig
        ik_cfg = ctx.config.robot_binding.ik
        action = CanonicalActionSequence.load_npz(ctx.layout.actions_npz)
        episode = _episode(ctx)
        hand_ids = list(action.hand_ids)
        timestamps = np.asarray(action["timestamp_s"], dtype=np.float64)
        frame_index = np.asarray(action["frame_index"], dtype=np.int64)
        T = int(timestamps.shape[0])

        frozen = self._load_frozen_rig(ctx)
        arm_records: dict[str, dict[str, Any]] = {}
        arms: dict[str, Any] = {}

        # ---- pass 1: build both arms, then resolve the two bases together ---
        import mujoco

        arm_ctx: dict[str, dict[str, Any]] = {}
        for side in ("right", "left"):
            arm_cfg = cfg.arm_for_side(side)
            h = hand_ids.index(side)
            robot_config, robot = build_robot_for(
                arm_cfg.robot_id,
                repo_root=ctx.repo_root,
                mjcf_uri=ctx.config.case.get("mjcf_uri"),
                ik_settings={
                    "position_cost": ik_cfg.position_cost,
                    "orientation_cost": ik_cfg.orientation_cost,
                    "lm_damping": ik_cfg.lm_damping,
                    "iterations": ik_cfg.iterations,
                    "tolerance": ik_cfg.tolerance,
                    "qp_backend": ik_cfg.qp_backend,
                },
            )
            bodies = [
                name
                for name in (
                    mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                    for bid in range(robot.model.nbody)
                )
                if name and name != "world"
            ]
            arm_ctx[side] = {
                "arm_cfg": arm_cfg,
                "hand_index": h,
                "robot_config": robot_config,
                "robot": robot,
                "T_C_TCP": np.asarray(action["T_C_TCP"], dtype=np.float64)[:, h],
                "valid": np.asarray(action["valid"], dtype=bool)[:, h],
                "width": np.asarray(action["gripper_width_m"], dtype=np.float64)[:, h],
                "bodies": bodies,
                "joint_names": list(robot_config.joint_names),
                "gripper_joint_names": list(robot_config.gripper_joint_names),
                "eef_body": robot_config.eef_body_name,
            }
        resolved, joint_search_report = self._resolve_bases(ctx, episode, action, arm_ctx, frozen)

        # ---- pass 2: solve each arm and publish its own trajectory ----------
        for side in ("right", "left"):
            entry = arm_ctx[side]
            arm_cfg = entry["arm_cfg"]
            robot = entry["robot"]
            robot_config = entry["robot_config"]
            h = entry["hand_index"]
            T_C_TCP = entry["T_C_TCP"]
            valid = entry["valid"]
            width = entry["width"]
            T_C_B, base_source, base_notes, search_payload = resolved[side]

            arm_result = solve_trajectory(
                robot=robot,
                robot_config=robot_config,
                ik=ik_cfg,
                targets_B=np.einsum("ij,tjk->tik", invert_transform(T_C_B), T_C_TCP),
                valid=valid,
                width_m=width,
                timestamps=timestamps,
                gripper_width_min_m=float(ctx.config.robot_binding.gripper_width_min_m),
            )
            trajectory = RobotTrajectory(
                {
                    "timestamp_s": timestamps,
                    "frame_index": frame_index,
                    "qpos": arm_result.qpos,
                    "qvel": arm_result.qvel,
                    "gripper_qpos": arm_result.gripper_qpos,
                    "T_B_TCP_target": arm_result.T_B_TCP_target[:, None, :, :],
                    "T_B_TCP_fk": arm_result.T_B_TCP_fk[:, None, :, :],
                    "position_error_m": arm_result.position_error_m[:, None],
                    "rotation_error_rad": arm_result.rotation_error_rad[:, None],
                    "ik_success": arm_result.ik_success[:, None],
                    "collision_code": arm_result.collision_code,
                    "joint_limit_hit": arm_result.joint_limit_hit,
                    "ik_iterations": arm_result.ik_iterations,
                    "solver_status": arm_result.solver_status,
                },
                meta={
                    "pose_variable": str(transform_name(Frame.B, Frame.TCP)),
                    "robot_id": robot_config.robot_id,
                    "arm_id": arm_cfg.arm_id,
                    "hand_id": side,
                    "T_C_B": T_C_B.tolist(),
                    "T_E_TCP": robot_config.T_E_TCP,
                    "mjcf": str(robot.mjcf_path),
                    "mjcf_sha256": robot_config.mjcf_sha256,
                    "ik_solver": f"mink/{ik_cfg.qp_backend}",
                    "status_registry": {},
                    "collision_registry": {"0": "none", "1": "self", "2": "scene"},
                    "hand_action_stats": {
                        "valid_frames": int(np.count_nonzero(valid)),
                        "width_min_m": round(float(np.nanmin(width[valid])), 6) if np.any(valid) else None,
                        "width_max_m": round(float(np.nanmax(width[valid])), 6) if np.any(valid) else None,
                    },
                },
            )
            trajectory.validate().raise_if_bad()
            arm_dir = ctx.layout.rig_arm_dir(arm_cfg.arm_id)
            arm_dir.mkdir(parents=True, exist_ok=True)
            result.add_output(trajectory.save_npz(arm_dir / "trajectory.npz"))

            arms[side] = {
                "arm_cfg": arm_cfg,
                "robot_config": robot_config,
                "robot": robot,
                "T_C_B": T_C_B,
                "base_source": base_source,
                "base_notes": base_notes,
                "arm_result": arm_result,
                "trajectory": trajectory,
                "hand_index": h,
                "valid": valid,
                "width": width,
                "search": search_payload,
            }
            arm_records[side] = {
                "arm_id": arm_cfg.arm_id,
                "hand_id": side,
                "robot_id": robot_config.robot_id,
                "base_source": base_source,
                "T_C_B": np.round(T_C_B, 9).tolist(),
                "base_notes": base_notes,
                "mjcf": str(robot.mjcf_path),
                "mjcf_sha256": robot_config.mjcf_sha256,
                "T_E_TCP": robot_config.T_E_TCP,
                "joint_names": list(robot_config.joint_names),
                "gripper_joint_names": list(robot_config.gripper_joint_names),
                "max_reach_m": robot_config.max_reach_m,
                "search": search_payload,
            }

        # ---- joint scene ---------------------------------------------------
        joint = self._joint_scene(ctx, episode, arms, result)

        # ---- metrics -------------------------------------------------------
        max_contacts = int(cfg.max_cross_arm_contacts)
        per_arm: dict[str, Any] = {}
        for side in ("right", "left"):
            arm = arms[side]
            arm_result = arm["arm_result"]
            metrics = arm_result.metrics(max_joint_jump_rad=float(ik_cfg.max_joint_jump_rad))
            metrics.update(
                {
                    "valid_frames": int(np.count_nonzero(arm["valid"])),
                    "ik_success_ratio": round(
                        metrics["ik_success"] / max(1, int(np.count_nonzero(arm["valid"]))), 6
                    ),
                    "visibility_mean_ratio": joint["visibility"][side]["mean"],
                    "visibility_min_ratio": joint["visibility"][side]["min"],
                    "fk_disagreement_max_m": joint["fk"][side]["max_m"],
                    "fk_disagreement_max_rad": joint["fk"][side]["max_rad"],
                    "width_above_model_max_frames": int(
                        np.count_nonzero(
                            arm["valid"]
                            & np.isfinite(arm["width"])
                            & (arm["width"] > float(ctx.config.robot_binding.gripper_width_max_m))
                        )
                    ),
                }
            )
            per_arm[side] = metrics

        jointly_valid = np.all(np.stack([arms[s]["valid"] for s in ("right", "left")]), axis=0)
        contacts = joint["contacts"]
        contact_frames = int(np.count_nonzero(contacts > max_contacts))
        gates = {
            "per_arm_ik": {
                "threshold": ">= 0.95 of each arm's valid frames",
                "right": per_arm["right"]["ik_success_ratio"],
                "left": per_arm["left"]["ik_success_ratio"],
                "pass": bool(per_arm["right"]["ik_success_ratio"] >= 0.95 and per_arm["left"]["ik_success_ratio"] >= 0.95),
            },
            "per_arm_joint_jump": {
                "threshold": f"<= {ik_cfg.max_joint_jump_rad} rad per frame",
                "right_frames": per_arm["right"]["joint_jump_frames"],
                "left_frames": per_arm["left"]["joint_jump_frames"],
                "pass": bool(
                    per_arm["right"]["joint_jump_frames"] == 0 and per_arm["left"]["joint_jump_frames"] == 0
                ),
            },
            "cross_arm_contacts": {
                "threshold": f"<= {max_contacts} contacts per frame",
                "frames_over_threshold": contact_frames,
                "max_contacts_per_frame": int(contacts.max()) if contacts.size else 0,
                "pass": bool(contact_frames == 0),
            },
            "joint_scene_fk": {
                "threshold": "<= 1e-6 m between independent FK and joint-scene FK",
                "right_max_m": joint["fk"]["right"]["max_m"],
                "left_max_m": joint["fk"]["left"]["max_m"],
                "pass": bool(
                    joint["fk"]["right"]["max_m"] <= 1e-6 and joint["fk"]["left"]["max_m"] <= 1e-6
                ),
            },
        }
        result.metrics.update(
            {
                "num_frames": T,
                "hand_ids": hand_ids,
                "arm_ids": [arm_records[s]["arm_id"] for s in ("right", "left")],
                "jointly_valid_frames": int(np.count_nonzero(jointly_valid)),
                "jointly_valid_ratio": round(float(np.mean(jointly_valid)), 6) if T else 0.0,
                "cross_arm_contact_frames": contact_frames,
                "cross_arm_contact_max": int(contacts.max()) if contacts.size else 0,
                "cross_arm_contact_pairs": joint["pair_union"],
                "per_arm": per_arm,
                "gates": gates,
                "not_evaluated": [
                    "visual_quality (no render/composite in B2; ADR-0022 B3)",
                    "grasp_success (no contact physics against the real object)",
                    "joint_two_base_search (per-arm bases frozen; pairwise search is next)",
                ],
            }
        )
        for side in ("right", "left"):
            for key in ("ik_success_ratio", "joint_jump_frames", "self_collision_frames", "visibility_min_ratio"):
                result.metrics[f"{side}_{key}"] = per_arm[side].get(key)

        # ---- evidence ------------------------------------------------------
        rig_payload = {
            "schema_version": "ego2robot.rig.v1",
            "episode_id": ctx.episode_id,
            "created_at": A.utc_now_iso(),
            "run_id": ctx.run_id,
            "config_hash": ctx.config_hash,
            "binding": {
                "hand_to_arm": {s: arm_records[s]["arm_id"] for s in ("right", "left")},
                "arm_to_robot": {s: arm_records[s]["robot_id"] for s in ("right", "left")},
                "note": (
                    "handedness sign is not a robot pose mirror; each arm is an independent "
                    "rigid placement of the same model (ADR-0022 decisions 2 and 4)"
                ),
            },
            "arms": arm_records,
            "action_sha256": A.file_sha256(ctx.layout.actions_npz),
            "joint_base_search": joint_search_report,
            "joint_scene": {
                "path": str(joint["scene_path"]),
                "sha256": A.file_sha256(joint["scene_path"]),
                "relative_T_W_BL": np.round(joint["relative_placement"], 9).tolist(),
                "world_frame": "right arm base frame (T_C_B of the right arm)",
                "collision_only": True,
                "nq": int(joint["scene"].model.nq),
                "nu": int(joint["scene"].model.nu),
                "nbody": int(joint["scene"].model.nbody),
            },
            "metrics": result.metrics,
            "limits": [
                "cross-arm contacts are model-vs-model contacts only; the manipulated object is "
                "not in the scene, so object-arm contacts are not detected",
                "visibility is an analytic projection proxy (link origins inside the image and in "
                "front of the camera), not a rendered occlusion test",
                "the two-arm scene contains collision geometry only: no RGB, masks or depth (B3)",
            ],
        }
        result.add_output(A.atomic_write_json(ctx.layout.rig_json, rig_payload))
        result.add_output(joint["contacts_path"])

        if cfg.write_previews:
            previews = self._write_previews(ctx, episode, action, arms, joint)
            for path in previews:
                result.add_output(path)
        if cfg.write_render:
            renders = self._write_render(ctx, episode, action, arms, result)
            for path in renders:
                result.add_output(path)

        result.payload.update(
            {"action": action, "arms": arms, "joint": joint, "payload": rig_payload}
        )
        return result

    # -- outputs -----------------------------------------------------------

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        res = ValidationResult()
        arms = result.payload.get("arms") or {}
        joint = result.payload.get("joint") or {}
        if sorted(arms) != ["left", "right"]:
            res.add("rig.no_arms", "rig stage produced no per-arm payload")
            return res
        for side in ("right", "left"):
            trajectory = arms[side]["trajectory"]
            res.extend(trajectory.validate())
            res.extend(
                trajectory.validate_numeric(
                    allow_nan_in=(
                        "T_B_TCP_target",
                        "T_B_TCP_fk",
                        "position_error_m",
                        "rotation_error_rad",
                    )
                )
            )
            success = np.asarray(trajectory["ik_success"])[:, 0]
            if not np.any(success):
                res.add(
                    "rig.zero_ik_success",
                    f"arm {side!r}: no frame solved IK; refusing to publish a two-arm rig "
                    "that looks complete",
                )
        if joint.get("fk"):
            worst = max(float(joint["fk"][s]["max_m"]) for s in ("right", "left"))
            if worst > 1e-6:
                # This is not a solver quality number: it means the generated joint
                # scene does not reproduce the independent per-arm FK, i.e. the
                # assembly (prefix/base/relative placement) is wrong.
                res.add(
                    "rig.joint_scene_disagreement",
                    f"joint-scene FK disagrees with the independent per-arm FK by "
                    f"{worst:.3e} m; the two-arm assembly is inconsistent",
                )
        for path in result.outputs:
            if not Path(path).is_file():
                res.add("rig.missing_output", f"declared output does not exist: {path}")
        return res

    # -- base -------------------------------------------------------------

    def _load_frozen_rig(self, ctx: StageContext) -> dict[str, Any] | None:
        if not ctx.layout.rig_json.is_file():
            return None
        try:
            payload = A.read_json(ctx.layout.rig_json)
        except Exception:  # noqa: BLE001 - a broken record is rebuilt, not trusted
            return None
        return payload if payload.get("schema_version") == "ego2robot.rig.v1" else None

    def _resolve_bases(
        self,
        ctx: StageContext,
        episode: EpisodeManifest,
        action: CanonicalActionSequence,
        arm_ctx: dict[str, dict[str, Any]],
        frozen: dict[str, Any] | None,
    ) -> tuple[dict[str, tuple[np.ndarray, str, list[str], dict[str, Any] | None]], dict[str, Any] | None]:
        """Resolve both bases: profile-fixed, frozen record, or pairwise joint search."""
        cfg = ctx.config.rig
        action_sha = A.file_sha256(ctx.layout.actions_npz)

        if cfg.base_mode == "fixed":
            resolved = {}
            for side in ("right", "left"):
                arm_cfg = arm_ctx[side]["arm_cfg"]
                resolved[side] = (
                    np.asarray(arm_cfg.T_C_B, dtype=np.float64),
                    "profile (rig.base_mode=fixed)",
                    ["explicit T_C_B from the pipeline profile"],
                    None,
                )
            return resolved, None

        if frozen is not None:
            frozen_sha = str(frozen.get("action_sha256") or "")
            frozen_arms = frozen.get("arms") or {}
            if frozen_sha == action_sha and frozen_arms.get("right") and frozen_arms.get("left"):
                resolved = {}
                for side in ("right", "left"):
                    resolved[side] = (
                        np.asarray(frozen_arms[side]["T_C_B"], dtype=np.float64),
                        "rig.json (frozen for this canonical action)",
                        ["reused because rig.base_mode is per_arm_search and the action hash matches"],
                        None,
                    )
                report = frozen.get("joint_base_search")
                return resolved, report
            result_warning = (
                "rig.json exists but its action_sha256 does not match the current canonical "
                "action; the bases are re-searched instead of being silently reused"
            )
            if ctx.log is not None:
                ctx.log.warning(result_warning)

        # --- pairwise joint search (ADR-0024) ------------------------------
        search_cfg = {
            side: BaseSearchConfig(
                lateral_levels=tuple(cfg.search.lateral_levels),
                forward_levels=tuple(cfg.search.forward_levels),
                vertical_levels=tuple(cfg.search.vertical_levels),
                pitch_deg=tuple(cfg.search.pitch_deg),
                yaw_deg=tuple(cfg.search.yaw_deg),
                roll_deg=tuple(cfg.search.roll_deg),
                min_camera_distance_m=float(cfg.search.min_camera_distance_m),
                min_base_distance_m=float(cfg.search.min_base_distance_m),
                max_reach_fraction=float(cfg.search.max_reach_fraction),
                max_keyframes=int(cfg.search.max_keyframes),
                reach_ratio_target=float(cfg.search.reach_ratio_target),
                reach_penalty_weight=float(cfg.search.reach_penalty_weight),
                ik_iterations=int(cfg.search.ik_iterations),
                ik_tolerance=float(cfg.search.ik_tolerance),
                lateral_sign=int(arm_ctx[side]["arm_cfg"].resolved_lateral_sign),
                min_forward_depth_m=float(cfg.search.min_forward_depth_m),
            )
            for side in ("right", "left")
        }
        keyframes = _joint_keyframes(action)
        bases, report = select_joint_bases(
            arms={side: arm_ctx[side] for side in ("right", "left")},
            search_cfg=search_cfg,
            top_k=int(cfg.search.top_k),
            max_cross_arm_contacts=int(cfg.max_cross_arm_contacts),
            min_base_separation_m=float(cfg.search.min_base_separation_m),
            joint_keyframes=keyframes,
            camera_K=episode.camera.K_array,
            camera_size=(episode.camera.width, episode.camera.height),
            ik_cfg=ctx.config.robot_binding.ik,
            timestamps=np.asarray(action["timestamp_s"], dtype=np.float64),
            gripper_width_min_m=float(ctx.config.robot_binding.gripper_width_min_m),
            max_joint_jump_rad=float(ctx.config.robot_binding.ik.max_joint_jump_rad),
            log=ctx.log,
        )
        report["action_sha256"] = action_sha
        resolved = {}
        for side in ("right", "left"):
            chosen = report["chosen"]
            resolved[side] = (
                bases[side],
                "pairwise joint base search (per-arm top_k x top_k)",
                [
                    f"right candidate {chosen['right_candidate']} x left candidate {chosen['left_candidate']}",
                    f"worst-arm keyframe visibility {chosen['min_visibility']:.3f}",
                    f"cross-arm contacts at keyframes {chosen['cross_arm_contacts_max']}",
                ],
                None,
            )
        return resolved, report

    # -- joint scene -------------------------------------------------------

    def _joint_scene(
        self, ctx: StageContext, episode: EpisodeManifest, arms: dict[str, Any], result: StageResult
    ) -> dict[str, Any]:
        right = arms["right"]
        left = arms["left"]
        if right["robot_config"].robot_id != left["robot_config"].robot_id:
            raise ValueError(
                "the joint scene generator currently assembles two instances of the *same* "
                f"model; got {right['robot_config'].robot_id!r} and "
                f"{left['robot_config'].robot_id!r} (ADR-0022 keeps ALOHA-class rigs for later)"
            )
        T_rel = relative_placement(right["T_C_B"], left["T_C_B"])
        scene_path = build_dual_scene(right["robot"].mjcf_path, T_rel, collision_only=True)

        import mujoco

        body_names = [
            mujoco.mj_id2name(right["robot"].model, mujoco.mjtObj.mjOBJ_BODY, bid)
            for bid in range(right["robot"].model.nbody)
        ]
        bodies = [n for n in body_names if n and n != "world"]
        scene = DualRigScene(
            model_path=scene_path,
            arm_bodies={side: bodies for side in ARM_PREFIX},
            arm_joint_names={
                side: list(arms[side]["robot_config"].joint_names) for side in ARM_PREFIX
            },
            arm_gripper_joint_names={
                side: list(arms[side]["robot_config"].gripper_joint_names) for side in ARM_PREFIX
            },
            arm_eef_body={
                side: arms[side]["robot_config"].eef_body_name for side in ARM_PREFIX
            },
        )

        T = int(right["arm_result"].num_frames)
        contacts = np.zeros(T, dtype=np.int32)
        pair_counts = np.zeros(T, dtype=np.int32)
        pair_union: set[tuple[str, str]] = set()
        fk = {side: {"max_m": 0.0, "max_rad": 0.0} for side in ARM_PREFIX}
        visibility = {side: {"mean": 0.0, "min": 1.0, "values": np.zeros(T)} for side in ARM_PREFIX}
        T_C_BR = right["T_C_B"]
        K = episode.camera.K_array
        width_px, height_px = episode.camera.width, episode.camera.height

        def _visible_fraction(points_C: np.ndarray) -> float:
            finite = np.isfinite(points_C).all(axis=1)
            if not finite.any():
                return 0.0
            P = points_C[finite]
            z = P[:, 2]
            u = K[0, 0] * P[:, 0] / np.where(np.abs(z) < 1e-9, np.nan, z) + K[0, 2]
            v = K[1, 1] * P[:, 1] / np.where(np.abs(z) < 1e-9, np.nan, z) + K[1, 2]
            ok = (z > 0.0) & (u >= 0) & (u < width_px) & (v >= 0) & (v < height_px)
            return float(np.count_nonzero(ok) / max(1, P.shape[0]))

        chain_positions: dict[str, np.ndarray] = {side: np.full((T, len(CHAIN), 3), np.nan) for side in ARM_PREFIX}
        for t in range(T):
            for side in ARM_PREFIX:
                scene.set_arm(
                    side,
                    arms[side]["arm_result"].qpos[t],
                    arms[side]["arm_result"].gripper_qpos[t],
                )
            scene.forward()
            n_contacts, pairs = scene.cross_arm_contacts()
            contacts[t] = n_contacts
            pair_counts[t] = len(pairs)
            pair_union.update(pairs)
            for side in ARM_PREFIX:
                expected = arms[side]["arm_result"].T_B_TCP_fk[t]
                if side == "left":
                    expected = T_rel @ expected
                actual = scene.arm_tcp(side, arms[side]["robot_config"].T_E_TCP_array)
                fk[side]["max_m"] = max(fk[side]["max_m"], float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])))
                fk[side]["max_rad"] = max(fk[side]["max_rad"], _rotation_error_rad(actual, expected))
                for i, body in enumerate(CHAIN):
                    T_W_body = scene.body_pose(ARM_PREFIX[side] + body)
                    chain_positions[side][t, i] = T_W_body[:3, 3]
            # visibility in the camera frame (world is the right arm base frame)
            points = np.vstack(
                [
                    (T_C_BR @ scene.body_pose(ARM_PREFIX[side] + body))[:3, 3]
                    for side in ARM_PREFIX
                    for body in CHAIN
                ]
            )
            for side in ARM_PREFIX:
                pts = np.vstack(
                    [
                        (T_C_BR @ scene.body_pose(ARM_PREFIX[side] + body))[:3, 3]
                        for body in CHAIN
                    ]
                )
                visibility[side]["values"][t] = _visible_fraction(pts)
            _ = points

        for side in ARM_PREFIX:
            vals = visibility[side]["values"]
            visibility[side]["mean"] = round(float(np.mean(vals)), 6)
            visibility[side]["min"] = round(float(np.min(vals)), 6)

        contacts_path = ctx.layout.rig_contacts_npz
        contacts_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            contacts_path,
            frame_index=np.asarray(arms["right"]["trajectory"]["frame_index"], dtype=np.int64),
            timestamp_s=np.asarray(arms["right"]["trajectory"]["timestamp_s"], dtype=np.float64),
            cross_arm_contacts=contacts,
            cross_arm_unique_pairs=pair_counts,
            fk_error_m_right=np.full(T, fk["right"]["max_m"], dtype=np.float64),
            fk_error_m_left=np.full(T, fk["left"]["max_m"], dtype=np.float64),
            visibility_right=visibility["right"]["values"].astype(np.float64),
            visibility_left=visibility["left"]["values"].astype(np.float64),
        )
        result.metrics["joint_scene"] = {
            "path": str(scene_path),
            "sha256": A.file_sha256(scene_path),
            "nq": int(scene.model.nq),
            "nbody": int(scene.model.nbody),
        }
        if fk["right"]["max_m"] > 1e-6 or fk["left"]["max_m"] > 1e-6:
            result.warnings.append(
                "joint-scene FK disagrees with the independent per-arm FK "
                f"({fk['right']['max_m']:.3e} m / {fk['left']['max_m']:.3e} m); the assembly "
                "is wrong, not the trajectory"
            )
        return {
            "scene": scene,
            "scene_path": scene_path,
            "relative_placement": T_rel,
            "contacts": contacts,
            "pair_counts": pair_counts,
            "pair_union": [list(p) for p in sorted(pair_union)],
            "fk": fk,
            "visibility": visibility,
            "contacts_path": contacts_path,
            "chain_positions": chain_positions,
        }

    # -- previews ----------------------------------------------------------

    # -- render (B3 first half, ADR-0025) ----------------------------------

    def _write_render(
        self,
        ctx: StageContext,
        episode: EpisodeManifest,
        action: CanonicalActionSequence,
        arms: dict[str, Any],
        result: StageResult,
    ) -> list[Path]:
        """Render both arms in one scene: RGB, owner map, per-arm masks, depth."""
        import json

        import imageio.v2 as imageio

        from ..vision.render_rig import RigRenderer

        cfg = ctx.config.rig
        outputs: list[Path] = []
        T_rel = relative_placement(arms["right"]["T_C_B"], arms["left"]["T_C_B"])
        scene_path = build_dual_scene(
            arms["right"]["robot"].mjcf_path,
            T_rel,
            collision_only=False,
            with_camera=True,
        )
        render_dir = ctx.layout.rig_dir / "render"
        render_dir.mkdir(parents=True, exist_ok=True)

        picks = np.linspace(0, int(action.num_frames) - 1, min(int(cfg.render_frames), int(action.num_frames)))
        picks = sorted({int(round(v)) for v in picks})
        frames: list[dict[str, Any]] = []
        depth_store: dict[str, np.ndarray] = {}
        with RigRenderer(
            scene_path,
            episode.camera,
            arms["right"]["T_C_B"],
            shadows=True,
        ) as renderer:
            for t in picks:
                renderer.set_arm(
                    "R",
                    arms["right"]["robot_config"].joint_names,
                    arms["right"]["arm_result"].qpos[t],
                    arms["right"]["robot_config"].gripper_joint_names,
                    arms["right"]["arm_result"].gripper_qpos[t],
                )
                renderer.set_arm(
                    "L",
                    arms["left"]["robot_config"].joint_names,
                    arms["left"]["arm_result"].qpos[t],
                    arms["left"]["robot_config"].gripper_joint_names,
                    arms["left"]["arm_result"].gripper_qpos[t],
                )
                frame_index = int(np.asarray(action["frame_index"])[t])
                frame = renderer.render(frame_index=frame_index)
                depth_store[f"depth_{frame_index:06d}"] = frame.depth_m
                owner_rgb = _owner_rgb(frame.owner)
                npz = render_dir / f"depth-frame-{frame_index:06d}.npz"
                np.savez_compressed(npz, depth_m=frame.depth_m, owner=frame.owner)
                outputs.append(npz)
                for name, array in (
                    (f"rgb-frame-{frame_index:06d}.png", frame.rgb),
                    (f"owner-frame-{frame_index:06d}.png", owner_rgb),
                    (f"mask-frame-{frame_index:06d}.png", (frame.arm_mask.astype(np.uint8) * 255)),
                ):
                    path = render_dir / name
                    imageio.imwrite(path, array)
                    outputs.append(path)
                frames.append(
                    {
                        "frame_index": frame_index,
                        "per_arm_pixels": frame.per_arm_pixels,
                        "depth_min_m": float(np.nanmin(frame.depth_m)) if np.isfinite(frame.depth_m).any() else None,
                        "depth_max_m": float(np.nanmax(frame.depth_m)) if np.isfinite(frame.depth_m).any() else None,
                    }
                )

        payload = {
            "schema_version": "ego2robot.rig.render.v1",
            "scene": str(scene_path),
            "scene_sha256": A.file_sha256(scene_path),
            "camera_frame": "C (right base world, T_W_C = inv(T_C_B_right))",
            "owner_map_legend": {"0": "background", "1": "right_arm", "2": "left_arm"},
            "frames": frames,
            "limits": [
                "robot-only render: no human, no background, no compositing",
                "owner map comes from MuJoCo segmentation IDs, not from RGB",
                "depth is the camera-frame metric depth of the rendered geometry",
            ],
        }
        summary = render_dir / "render.json"
        summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(summary)
        result.metrics["render"] = {
            "frames": len(frames),
            "scene": str(scene_path),
            "per_arm_pixels": {side: [f["per_arm_pixels"][side] for f in frames] for side in ("right", "left")},
        }
        _ = depth_store
        return outputs

    # -- render helper ------------------------------------------------------

    def _write_previews(
        self,
        ctx: StageContext,
        episode: EpisodeManifest,
        action: CanonicalActionSequence,
        arms: dict[str, Any],
        joint: dict[str, Any],
    ) -> list[Path]:
        import cv2

        from ..ingest.video import iter_frames
        from ..viz.frames_3d import (
            draw_camera_frustum,
            draw_frame_axes,
            draw_robot_arm,
            render_scene_figure,
        )
        from ..viz.hand_overlay import write_image, write_overlay_video

        cfg = ctx.config.rig
        outputs: list[Path] = []
        T = int(joint["contacts"].shape[0])
        previews = ctx.layout.previews_dir
        previews.mkdir(parents=True, exist_ok=True)

        video = ctx.layout.video_path
        if not video.is_file():
            from ..ingest.stage import discover_sources

            discovered, _hdf5, _camera = discover_sources(ctx.layout.root)
            video = discovered
        if video is None or not Path(video).is_file():
            return outputs

        K = episode.camera.K_array
        T_C_BR = arms["right"]["T_C_B"]
        widths = {
            side: {
                "min": float(np.nanmin(arms[side]["width"][arms[side]["valid"]]))
                if np.any(arms[side]["valid"])
                else float("nan"),
                "max": float(np.nanmax(arms[side]["width"][arms[side]["valid"]]))
                if np.any(arms[side]["valid"])
                else float("nan"),
            }
            for side in ARM_PREFIX
        }

        def _project(points: np.ndarray) -> np.ndarray:
            P = np.asarray(points, dtype=np.float64)
            z = P[:, 2]
            safe = np.where(np.abs(z) < 1e-9, np.nan, z)
            return np.stack([K[0, 0] * P[:, 0] / safe + K[0, 2], K[1, 1] * P[:, 1] / safe + K[1, 2]], axis=1)

        chain_world = joint["chain_positions"]
        frames: list[np.ndarray] = []
        for t, (_idx, ts, image_rgb) in enumerate(iter_frames(video, start=0, stop=T)):
            if t >= T:
                break
            canvas = np.ascontiguousarray(image_rgb[:, :, ::-1].copy())
            lines: list[str] = [
                f"frame {int(np.asarray(action['frame_index'])[t])}  t={ts:.3f}s  "
                f"cross-arm contacts {int(joint['contacts'][t])} (pairs {int(joint['pair_counts'][t])})"
            ]
            for side in ARM_PREFIX:
                colour = ARM_COLOUR[side]
                bgr = tuple(int(c) for c in colour[::-1])
                pts_W = chain_world[side][t]
                pts_C = np.vstack([(T_C_BR @ make_transform(np.eye(3), p))[:3, 3] for p in pts_W])
                uv = _project(pts_C)
                finite = np.isfinite(uv).all(axis=1)
                if finite.sum() >= 2:
                    pts = [(int(round(u)), int(round(v))) for u, v in uv[finite]]
                    cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, bgr, 2, cv2.LINE_AA)
                # FK TCP marker
                fk_T = arms[side]["arm_result"].T_B_TCP_fk[t]
                if side == "left":
                    fk_T = joint["relative_placement"] @ fk_T
                if np.isfinite(fk_T).all():
                    p_fk = (T_C_BR @ fk_T)[:3, 3]
                    uv_fk = _project(p_fk[None, :])[0]
                    if np.isfinite(uv_fk).all():
                        o = (int(round(uv_fk[0])), int(round(uv_fk[1])))
                        cv2.drawMarker(canvas, o, bgr, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                # target TCP axes
                T_target = np.asarray(action["T_C_TCP"], dtype=np.float64)[t, arms[side]["hand_index"]]
                if np.isfinite(T_target).all():
                    length = 0.04
                    origin = (T_C_BR @ T_target)[:3, 3]
                    tips = [origin] + [
                        origin + (T_C_BR @ T_target)[:3, i] * length for i in range(3)
                    ]
                    uv_t = _project(np.vstack(tips))
                    if np.isfinite(uv_t).all():
                        o = (int(round(uv_t[0, 0])), int(round(uv_t[0, 1])))
                        for i in range(3):
                            cv2.arrowedLine(
                                canvas,
                                o,
                                (int(round(uv_t[i + 1, 0])), int(round(uv_t[i + 1, 1]))),
                                tuple(int(c) for c in AXIS_RGB[i][::-1]),
                                2,
                                cv2.LINE_AA,
                                tipLength=0.3,
                            )
                lines.append(
                    f"{side}: width {widths[side]['min'] * 1000:.0f}-{widths[side]['max'] * 1000:.0f} mm  "
                    f"vis {joint['visibility'][side]['values'][t]:.2f}"
                )
            y = 26
            for text in lines:
                cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                y += 24
            frames.append(np.ascontiguousarray(canvas[:, :, ::-1]))

        if frames:
            vp = previews / "bimanual-projection.mp4"
            fps = 1.0 / float(np.median(np.diff(np.asarray(action["timestamp_s"])))) if T > 1 else 30.0
            write_overlay_video(vp, frames, fps=fps)
            outputs.append(vp)
            picks = set(np.linspace(0, T - 1, min(int(cfg.sample_stills), T)).astype(int).tolist())
            if joint["pair_counts"].size:
                picks.add(int(np.argmax(joint["pair_counts"])))
            tcp_dist = np.linalg.norm(
                np.asarray(action["tcp_position_C_m"], dtype=np.float64)[:, 0]
                - np.asarray(action["tcp_position_C_m"], dtype=np.float64)[:, 1],
                axis=1,
            )
            if np.isfinite(tcp_dist).any():
                picks.add(int(np.nanargmin(tcp_dist)))
            for t in sorted(picks):
                p = previews / f"bimanual-projection-frame-{int(np.asarray(action['frame_index'])[t]):06d}.png"
                write_image(p, frames[t])
                outputs.append(p)

        # ---- 3-D view ------------------------------------------------------
        mid = int(np.argmin(np.linalg.norm(
            np.asarray(action["tcp_position_C_m"], dtype=np.float64)[:, 0]
            - np.asarray(action["tcp_position_C_m"], dtype=np.float64)[:, 1],
            axis=1,
        )))

        def _draw(ax: Any) -> None:
            draw_camera_frustum(ax, np.eye(4), label="camera C")
            for side in ARM_PREFIX:
                positions = {
                    body: chain_world[side][mid, i] for i, body in enumerate(CHAIN)
                }
                draw_robot_arm(
                    ax,
                    positions,
                    order=CHAIN,
                    colour="#1f77b4" if side == "right" else "#ff7f0e",
                )
                T_target = np.asarray(action["T_C_TCP"], dtype=np.float64)[mid, arms[side]["hand_index"]]
                if np.isfinite(T_target).all():
                    draw_frame_axes(ax, T_target, length=0.04, label=f"{side} target")

        p3d = render_scene_figure(
            previews / "bimanual-rig-3d.png",
            title=(
                f"two-arm rig in frame C ({ctx.episode_id}, frame "
                f"{int(np.asarray(action['frame_index'])[mid])})"
            ),
            draw=_draw,
            point_sets=[chain_world[s].reshape(-1, 3) for s in ARM_PREFIX],
        )
        outputs.append(p3d)
        return outputs
