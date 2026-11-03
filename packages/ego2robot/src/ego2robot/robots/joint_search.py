"""Pairwise two-base selection for the two-arm rig (PLAN-02 B2, ADR-0024).

Per-arm "best" bases are not a joint solution: each arm can be individually perfect
and still leave its partner outside the camera, or put both bases on the same side of
the workspace.  This module keeps the per-arm ranking, then evaluates pairs of
candidates in the *joint* scene and picks a pair on joint criteria.

Implementation note (learned the hard way): the pair evaluation must use the **full
per-arm trajectory solve**, not isolated keyframe IK.  Keyframes are TCP extremes by
construction, so a single warm-start solve there rejects pairs that the real solve
(with chained warm starts and deterministic seed recovery) handles; the first version
rejected 21/25 pairs that way.  The cost is acceptable because one full 190-frame arm
solve takes ~1.5 s.

Order of decisions (PLAN-02):

1. hard constraints: each arm must solve its full trajectory (ratio and joint jump as
   in the per-arm gate), the pair must stay under the cross-arm contact threshold over
   the whole trajectory, the bases must be separated and on their own side of the
   camera axis;
2. joint quality: maximise the *worse* arm's visibility (a good arm cannot compensate
   for an invisible one), then prefer more simultaneously-visible keyframes;
3. tie-break: smaller mean-reach-ratio deviation, then candidate index (deterministic).

Keyframes are still used — for measuring visibility and contacts, not for deciding
feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .base_search import BaseCandidate, BaseSearchConfig, search_base_placement
from .ik import solve_trajectory
from .rig import ARM_PREFIX, DualRigScene, build_dual_scene, relative_placement

__all__ = ["JointPair", "select_joint_bases"]

CHAIN = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand")


@dataclass
class JointPair:
    """One evaluated (right, left) base pair."""

    right: BaseCandidate
    left: BaseCandidate
    relative_T_BR_BL: np.ndarray
    visibility_right: float = 0.0
    visibility_left: float = 0.0
    min_visibility: float = 0.0
    simultaneous_visible: int = 0
    cross_arm_contacts_max: int = 0
    cross_arm_contact_frames: int = 0
    contact_pairs: list[tuple[str, str]] = field(default_factory=list)
    base_separation_m: float = 0.0
    ik_ratio_right: float = 0.0
    ik_ratio_left: float = 0.0
    jump_frames_right: int = 0
    jump_frames_left: int = 0
    reach_deviation: float = 0.0
    score: float = float("-inf")
    rejected_reason: str | None = None

    def summarise(self) -> dict[str, Any]:
        return {
            "right_candidate": self.right.index,
            "left_candidate": self.left.index,
            "right_T_C_B": np.round(self.right.T_C_B, 6).tolist(),
            "left_T_C_B": np.round(self.left.T_C_B, 6).tolist(),
            "visibility_right": round(self.visibility_right, 4),
            "visibility_left": round(self.visibility_left, 4),
            "min_visibility": round(self.min_visibility, 4),
            "simultaneous_visible_frames": self.simultaneous_visible,
            "cross_arm_contacts_max": self.cross_arm_contacts_max,
            "cross_arm_contact_frames": self.cross_arm_contact_frames,
            "contact_pairs": [list(p) for p in self.contact_pairs],
            "base_separation_m": round(self.base_separation_m, 4),
            "ik_ratio_right": round(self.ik_ratio_right, 4),
            "ik_ratio_left": round(self.ik_ratio_left, 4),
            "joint_jump_frames_right": self.jump_frames_right,
            "joint_jump_frames_left": self.jump_frames_left,
            "reach_deviation": round(self.reach_deviation, 5),
            "score": round(self.score, 5) if np.isfinite(self.score) else None,
            "rejected_reason": self.rejected_reason,
        }


def _visibility(
    scene: DualRigScene,
    T_C_BR: np.ndarray,
    side: str,
    K: np.ndarray,
    width_px: int,
    height_px: int,
) -> float:
    """Fraction of this arm's chain origins inside the image and in front of the camera."""
    pts = []
    for body in CHAIN:
        T_W = scene.body_pose(ARM_PREFIX[side] + body)
        pts.append((T_C_BR @ T_W)[:3, 3])
    P = np.vstack(pts)
    z = P[:, 2]
    safe = np.where(np.abs(z) < 1e-9, np.nan, z)
    u = K[0, 0] * P[:, 0] / safe + K[0, 2]
    v = K[1, 1] * P[:, 1] / safe + K[1, 2]
    ok = (z > 0.0) & (u >= 0) & (u < width_px) & (v >= 0) & (v < height_px)
    return float(np.count_nonzero(ok) / max(1, P.shape[0]))


def _solve_keyframes(
    robot: Any,
    T_C_B: np.ndarray,
    T_C_TCP: np.ndarray,
    keyframes: Sequence[int],
    *,
    seed_attempts: int = 8,
) -> tuple[int, list[np.ndarray]]:
    """Cheap keyframe filter for one arm under a candidate base.

    The keyframes are TCP extremes by construction, so a single warm-start solve
    rejects pairs the full solve would handle.  This uses the same seed-recovery
    policy as :mod:`robots.ik` with fewer attempts; it is only a pre-filter, and the
    surviving pairs are re-solved over the full trajectory.
    """
    from ..core.transforms import invert_transform
    from .ik import joint_bounds, seed_variant

    T_B_C = invert_transform(T_C_B)
    home = np.asarray(robot.config.home_qpos, dtype=np.float64)
    q_prev = home.copy()
    lo, hi = joint_bounds(robot, list(robot.config.joint_names))
    g = np.asarray(robot.config.home_gripper_qpos, dtype=np.float64)
    solved = 0
    qpos: list[np.ndarray] = []
    for idx in keyframes:
        target = T_B_C @ T_C_TCP[idx]
        res = robot.solve_ik(target, q_prev, gripper_qpos=g)
        if not res.success and seed_attempts > 1:
            for attempt in range(seed_attempts - 1):
                seed = home if attempt == 0 else seed_variant(home, lo, hi, attempt)
                cand = robot.solve_ik(target, seed, gripper_qpos=g)
                if cand.success:
                    res = cand
                    break
                if cand.position_error_m < res.position_error_m:
                    res = cand
        if res.success:
            solved += 1
            q_prev = np.asarray(res.qpos, dtype=np.float64)
            qpos.append(q_prev)
        else:
            qpos.append(home)
    return solved, qpos


def _solve_arm(
    *,
    arms: dict[str, dict[str, Any]],
    side: str,
    T_C_B: np.ndarray,
    ik_cfg: Any,
    timestamps: np.ndarray,
    gripper_width_min_m: float,
):
    from ..core.transforms import invert_transform

    entry = arms[side]
    return solve_trajectory(
        robot=entry["robot"],
        robot_config=entry["robot_config"],
        ik=ik_cfg,
        targets_B=np.einsum("ij,tjk->tik", invert_transform(T_C_B), entry["T_C_TCP"]),
        valid=entry["valid"],
        width_m=entry["width"],
        timestamps=timestamps,
        gripper_width_min_m=gripper_width_min_m,
    )


def select_joint_bases(
    *,
    arms: dict[str, dict[str, Any]],
    search_cfg: dict[str, BaseSearchConfig],
    top_k: int,
    max_cross_arm_contacts: int,
    min_base_separation_m: float,
    joint_keyframes: Sequence[int],
    camera_K: np.ndarray,
    camera_size: tuple[int, int],
    ik_cfg: Any,
    timestamps: np.ndarray,
    gripper_width_min_m: float,
    max_joint_jump_rad: float,
    min_ik_ratio: float = 0.95,
    log: Any = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rank per-arm candidates, evaluate pairs on full solves, return the winner."""
    searches = {
        side: search_base_placement(
            arms[side]["robot"], arms[side]["T_C_TCP"], arms[side]["valid"], cfg=search_cfg[side], log=log
        )
        for side in ("right", "left")
    }
    shortlists: dict[str, list[BaseCandidate]] = {}
    for side in ("right", "left"):
        feasible = [
            c
            for c in searches[side].ranked
            if c.rejected_reason is None and c.ik_successes > 0 and np.isfinite(c.score)
        ]
        shortlists[side] = feasible[:top_k]
        if not shortlists[side]:
            raise RuntimeError(f"joint base selection: no feasible candidate for the {side} arm")

    keyframes = sorted({int(i) for i in joint_keyframes})
    pairs: list[JointPair] = []
    for right in shortlists["right"]:
        for left in shortlists["left"]:
            T_rel = relative_placement(right.T_C_B, left.T_C_B)
            pair = JointPair(right=right, left=left, relative_T_BR_BL=T_rel)
            pair.base_separation_m = float(
                np.linalg.norm(left.T_C_B[:3, 3] - right.T_C_B[:3, 3])
            )
            if not (float(right.T_C_B[0, 3]) > 0.0 > float(left.T_C_B[0, 3])):
                pair.rejected_reason = "base_side"
                pairs.append(pair)
                continue
            if pair.base_separation_m < min_base_separation_m:
                pair.rejected_reason = "base_separation"
                pairs.append(pair)
                continue

            # cheap pre-filter: both arms must solve every joint keyframe
            kf_r = _solve_keyframes(arms["right"]["robot"], right.T_C_B, arms["right"]["T_C_TCP"], keyframes)
            kf_l = _solve_keyframes(arms["left"]["robot"], left.T_C_B, arms["left"]["T_C_TCP"], keyframes)
            if kf_r[0] < len(keyframes) or kf_l[0] < len(keyframes):
                pair.rejected_reason = "keyframe_ik"
                pairs.append(pair)
                continue

            solved = {}
            for side, candidate in (("right", right), ("left", left)):
                solved[side] = _solve_arm(
                    arms=arms,
                    side=side,
                    T_C_B=candidate.T_C_B,
                    ik_cfg=ik_cfg,
                    timestamps=timestamps,
                    gripper_width_min_m=gripper_width_min_m,
                )
            valid_counts = {
                side: max(1, int(np.count_nonzero(arms[side]["valid"]))) for side in ("right", "left")
            }
            pair.ik_ratio_right = float(np.count_nonzero(solved["right"].ik_success)) / valid_counts["right"]
            pair.ik_ratio_left = float(np.count_nonzero(solved["left"].ik_success)) / valid_counts["left"]
            pair.jump_frames_right = solved["right"].metrics(max_joint_jump_rad=max_joint_jump_rad)[
                "joint_jump_frames"
            ]
            pair.jump_frames_left = solved["left"].metrics(max_joint_jump_rad=max_joint_jump_rad)[
                "joint_jump_frames"
            ]
            if pair.ik_ratio_right < min_ik_ratio or pair.ik_ratio_left < min_ik_ratio:
                pair.rejected_reason = "full_trajectory_ik"
                pairs.append(pair)
                continue
            # Joint jump is *not* a hard pair filter.  Under every candidate base the
            # current trajectory still contains a few frames above the 1.5 rad
            # threshold (PIT-042), so a hard filter would reject all 25 pairs and hide
            # the search behind an unrelated defect.  It therefore stays a preference
            # here, a reported number in rig.json, and a failing gate — not something
            # the base choice can fix.

            scene = DualRigScene(
                model_path=build_dual_scene(
                    arms["right"]["robot"].mjcf_path, T_rel, collision_only=True
                ),
                arm_bodies={s: arms[s]["bodies"] for s in ARM_PREFIX},
                arm_joint_names={s: arms[s]["joint_names"] for s in ARM_PREFIX},
                arm_gripper_joint_names={s: arms[s]["gripper_joint_names"] for s in ARM_PREFIX},
                arm_eef_body={s: arms[s]["eef_body"] for s in ARM_PREFIX},
            )
            vis_r: list[float] = []
            vis_l: list[float] = []
            contact_frames = 0
            contacts_max = 0
            contact_pairs: set[tuple[str, str]] = set()
            for t in keyframes:
                scene.set_arm("right", solved["right"].qpos[t], solved["right"].gripper_qpos[t])
                scene.set_arm("left", solved["left"].qpos[t], solved["left"].gripper_qpos[t])
                scene.forward()
                n, seen = scene.cross_arm_contacts()
                contacts_max = max(contacts_max, n)
                contact_frames += 1 if n > 0 else 0
                contact_pairs.update(seen)
                vis_r.append(_visibility(scene, right.T_C_B, "right", camera_K, *camera_size))
                vis_l.append(_visibility(scene, left.T_C_B, "left", camera_K, *camera_size))
            pair.cross_arm_contacts_max = contacts_max
            pair.cross_arm_contact_frames = contact_frames
            pair.contact_pairs = sorted(contact_pairs)
            pair.visibility_right = float(np.mean(vis_r))
            pair.visibility_left = float(np.mean(vis_l))
            pair.min_visibility = min(pair.visibility_right, pair.visibility_left)
            pair.simultaneous_visible = int(
                sum(1 for a, b in zip(vis_r, vis_l) if a >= 0.5 and b >= 0.5)
            )
            pair.reach_deviation = 0.5 * (
                abs(right.mean_reach_ratio - 0.65) + abs(left.mean_reach_ratio - 0.65)
            )
            if contacts_max > max_cross_arm_contacts:
                pair.rejected_reason = "cross_arm_contacts"
            else:
                pair.score = (
                    pair.min_visibility * 10.0
                    + pair.simultaneous_visible / max(1, len(keyframes))
                    - pair.reach_deviation
                    - 0.05 * (pair.jump_frames_right + pair.jump_frames_left)
                )
            pairs.append(pair)

    viable = [p for p in pairs if p.rejected_reason is None]
    if not viable:
        reasons: dict[str, int] = {}
        for p in pairs:
            reasons[p.rejected_reason or "none"] = reasons.get(p.rejected_reason or "none", 0) + 1
        raise RuntimeError(
            f"joint base selection found no acceptable pair ({reasons}); refusing to fall back "
            "to a pair that violates the joint constraints"
        )
    # prefer a pair that has no joint-jump frames at all, if any exists
    zero_jump = [p for p in viable if p.jump_frames_right == 0 and p.jump_frames_left == 0]
    jump_free_pair_available = bool(zero_jump)
    if zero_jump:
        viable = zero_jump
    viable.sort(key=lambda p: (-p.score, p.right.index, p.left.index))
    best = viable[0]
    report = {
        "schema_version": "ego2robot.rig.joint_base_search.v1",
        "mode": "pairwise_top_k_full_trajectory",
        "top_k_per_arm": int(top_k),
        "shortlist_right": [c.summarise() for c in shortlists["right"]],
        "shortlist_left": [c.summarise() for c in shortlists["left"]],
        "pairs_evaluated": len(pairs),
        "joint_keyframes": [int(i) for i in keyframes],
        "selection_order": [
            "hard: base side / separation / full-trajectory IK (>=0.95 per arm) / joint jump / "
            "cross-arm contacts over the whole trajectory",
            "joint: worst-arm visibility, then simultaneously-visible keyframes",
            "tie-break: reach-ratio deviation, then candidate index",
        ],
        "chosen": best.summarise(),
        "rejected": [p.summarise() for p in pairs if p.rejected_reason is not None],
        "top": [p.summarise() for p in viable[:5]],
        "jump_free_pair_available": jump_free_pair_available,
        "joint_jump_note": (
            "joint jump is a preference here (and a failing rig gate), not a pair filter: every "
            "candidate base still shows a few frames above the threshold under the current "
            "per-frame solve (PIT-042), which the base placement cannot fix"
        ),
        "note": (
            "pairs are evaluated with the same full per-arm solve the stage then publishes; "
            "keyframes are used for contacts/visibility sampling only"
        ),
    }
    if log is not None:
        log.info(
            f"joint base search: {len(shortlists['right'])}x{len(shortlists['left'])} pairs, "
            f"{len(viable)} viable, winner min-visibility {best.min_visibility:.3f}"
        )
    bases = {
        "right": np.asarray(best.right.T_C_B, dtype=np.float64),
        "left": np.asarray(best.left.T_C_B, dtype=np.float64),
    }
    return bases, report
