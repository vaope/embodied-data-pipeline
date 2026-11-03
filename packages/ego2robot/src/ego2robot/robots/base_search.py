"""Robot base placement: fixed (MVP) and paper-style grid search (M4).

System design 8.2.  Two strategies:

``fixed``
    ``T_C_B`` comes from the episode's ``base.json`` / profile.  This is the MVP
    path and the baseline every automatic search must beat.

``grid_search``
    [PAPER] appendix A.4: candidates are generated in the camera frame with
    offsets scaled by the robot reach ``r`` —

    * lateral:  ``r x {0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2}`` (7, sign per arm)
    * forward:  ``r x {-0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9}`` (7)
    * vertical: ``r x {0.4, 0.2, 0.0, -0.2, -0.4}`` (5)
    * orientation: pitch {30, 45, 60} deg x yaw {-45, -20, 0, 20, 45} deg
      x roll {-15, 0, 15} deg

    candidates closer than ``0.20 m`` to the camera, further than ``0.9 r`` from
    the base, or closer than ``0.08 m`` to the base are discarded; surviving
    candidates are scored by

    ``S = FR - 5.0 * |mean_reach_ratio - 0.65|``

    with up to 20 keyframes, Mink IK at 100 iterations and a ``1e-5`` threshold.

Keyframe selection is a separate module-level function on purpose: the plan
forbids hard-coding "every N frames" inside the search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.frames import Frame, transform_name
from ..core.transforms import (
    compose_transforms,
    exp_so3,
    invert_transform,
    make_transform,
    transform_points,
)

__all__ = [
    "BaseCandidate",
    "BaseSearchConfig",
    "BaseSearchResult",
    "FixedBaseResult",
    "candidate_orientations",
    "candidate_offsets",
    "load_fixed_base",
    "score_candidates",
    "search_base_placement",
    "select_keyframes",
]


@dataclass
class BaseSearchConfig:
    """Grid-search settings mirroring the paper's appendix A.4."""

    lateral_levels: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2)
    forward_levels: tuple[float, ...] = (-0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
    vertical_levels: tuple[float, ...] = (0.4, 0.2, 0.0, -0.2, -0.4)
    pitch_deg: tuple[float, ...] = (30.0, 45.0, 60.0)
    yaw_deg: tuple[float, ...] = (-45.0, -20.0, 0.0, 20.0, 45.0)
    roll_deg: tuple[float, ...] = (-15.0, 0.0, 15.0)
    min_camera_distance_m: float = 0.20
    min_base_distance_m: float = 0.08
    max_reach_fraction: float = 0.9
    max_keyframes: int = 20
    reach_ratio_target: float = 0.65
    reach_penalty_weight: float = 5.0
    ik_iterations: int = 100
    ik_tolerance: float = 1e-5
    lateral_sign: int = 1
    """``+1`` or ``-1`` — the paper flips the lateral axis per arm."""
    min_forward_depth_m: float = 0.0
    """[ENGINEERING] minimum camera-forward depth (``T_C_B[2,3]``) of the base.

    The paper's constraint is a distance from the camera, which a base *beside* the
    camera already satisfies while sitting almost in the camera plane.  The bimanual
    rig sets this so both bases stay in front of the camera; the single-arm default
    of 0 keeps the reviewed M3 behaviour unchanged (ADR-0023).
    """


@dataclass
class FixedBaseResult:
    """A pinned base placement plus where it came from."""

    T_C_B: NDArray[np.float64]
    source: str
    notes: list[str] = field(default_factory=list)

    def to_json(self, **extra: Any) -> dict[str, Any]:
        payload = {
            "schema_version": "ego2robot.base.v1",
            "mode": "fixed",
            "pose_variable": str(transform_name(Frame.C, Frame.B)),
            "T_C_B": self.T_C_B.tolist(),
            "source": self.source,
            "notes": list(self.notes),
        }
        payload.update(extra)
        return payload


@dataclass
class BaseCandidate:
    """One candidate placement and its keyframe evaluation."""

    index: int
    T_C_B: NDArray[np.float64]
    offset_reach_units: tuple[float, float, float]
    orientation_deg: tuple[float, float, float]
    feasibility_rate: float = 0.0
    mean_reach_ratio: float = 0.0
    score: float = float("-inf")
    rejected_reason: str | None = None
    ik_successes: int = 0
    keyframes_total: int = 0
    mean_position_error_m: float = 0.0
    mean_rotation_error_rad: float = 0.0

    def summarise(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "offset_reach_units": [round(v, 6) for v in self.offset_reach_units],
            "orientation_deg": [round(v, 3) for v in self.orientation_deg],
            "feasibility_rate": round(self.feasibility_rate, 6),
            "mean_reach_ratio": round(self.mean_reach_ratio, 6),
            "score": round(self.score, 6) if np.isfinite(self.score) else None,
            "rejected_reason": self.rejected_reason,
            "ik_successes": self.ik_successes,
            "keyframes_total": self.keyframes_total,
            "mean_position_error_m": round(self.mean_position_error_m, 9),
            "mean_rotation_error_rad": round(self.mean_rotation_error_rad, 9),
        }


@dataclass
class BaseSearchResult:
    """Outcome of a grid search: the winner, the ranking and full bookkeeping."""

    T_C_B: NDArray[np.float64]
    best: BaseCandidate
    ranked: list[BaseCandidate]
    total_candidates: int
    rejected_by_camera_distance: int
    rejected_by_reach: int
    keyframe_indices: list[int]
    config: BaseSearchConfig
    notes: list[str] = field(default_factory=list)
    rejected_by_camera_depth: int = 0

    def to_json(self, **extra: Any) -> dict[str, Any]:
        payload = {
            "schema_version": "ego2robot.base.v1",
            "mode": "grid_search",
            "pose_variable": str(transform_name(Frame.C, Frame.B)),
            "T_C_B": self.T_C_B.tolist(),
            "total_candidates": self.total_candidates,
            "rejected_by_camera_distance": self.rejected_by_camera_distance,
            "rejected_by_camera_depth": self.rejected_by_camera_depth,
            "rejected_by_reach": self.rejected_by_reach,
            "keyframe_indices": list(self.keyframe_indices),
            "best": self.best.summarise(),
            "top": [c.summarise() for c in self.ranked[:10]],
            "score_formula": "FR - reach_penalty_weight * abs(mean_reach_ratio - 0.65)",
            "notes": list(self.notes),
        }
        payload.update(extra)
        return payload


# ---------------------------------------------------------------------------
# Fixed base
# ---------------------------------------------------------------------------


def load_fixed_base(
    *,
    episode_T_C_B: Sequence[Sequence[float]] | NDArray[np.floating[Any]] | None,
    profile_T_C_B: Sequence[Sequence[float]] | NDArray[np.floating[Any]] | None = None,
) -> FixedBaseResult:
    """Resolve ``T_C_B`` for the MVP path, declaring where it came from."""
    for label, candidate in (("episode", episode_T_C_B), ("profile", profile_T_C_B)):
        if candidate is None:
            continue
        T = np.asarray(candidate, dtype=np.float64).reshape(4, 4)
        if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError(f"{label} T_C_B bottom row must be [0, 0, 0, 1]")
        return FixedBaseResult(
            T_C_B=T,
            source=label,
            notes=[
                "fixed base placement: operator supplied (system-design.md 8.2 MVP path)",
                f"pose of frame B expressed in frame C ({Frame.B.value} in {Frame.C.value})",
            ],
        )
    raise ValueError(
        "no T_C_B available. The MVP profile requires an explicit fixed base "
        "(episode source/base.json or the robot profile). Refusing to invent one "
        "(system-design.md 8.2, AGENTS.md rule 10)."
    )


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------


def select_keyframes(
    T_C_TCP: NDArray[np.floating[Any]],
    valid: NDArray[np.bool_] | None = None,
    *,
    max_keyframes: int = 20,
) -> list[int]:
    """Pick keyframes covering position extrema and orientation change extrema.

    The plan explicitly forbids a "every N frames" sampler here: a placement that
    only satisfies a uniformly subsampled set can still fail at the trajectory's
    spatial and rotational extremes.
    """
    T = np.asarray(T_C_TCP, dtype=np.float64)
    n = T.shape[0]
    if n == 0:
        return []
    mask = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).copy()
    if not mask.any():
        mask[:] = True
    idx_valid = np.nonzero(mask)[0]

    pos = T[:, :3, 3]
    rotvec = np.zeros((n, 3), dtype=np.float64)
    ref = T[idx_valid[0], :3, :3]
    from ..core.transforms import log_so3

    for t in idx_valid:
        rotvec[t] = log_so3(ref.T @ T[t, :3, :3])

    scores: dict[int, float] = {}
    for axis in range(3):
        col = pos[idx_valid, axis]
        scores[int(idx_valid[int(np.argmin(col))])] = 1e6
        scores[int(idx_valid[int(np.argmax(col))])] = 1e6
    for axis in range(3):
        col = rotvec[idx_valid, axis]
        scores[int(idx_valid[int(np.argmin(col))])] = 1e6
        scores[int(idx_valid[int(np.argmax(col))])] = 1e6

    # add the endpoints and the centroid-nearest frame, then fill up to the cap
    for t in (int(idx_valid[0]), int(idx_valid[-1])):
        scores.setdefault(t, 1e5)
    centroid = pos[idx_valid].mean(axis=0)
    scores.setdefault(int(idx_valid[int(np.argmin(np.linalg.norm(pos[idx_valid] - centroid, axis=1)))]), 1e4)

    remaining = [int(t) for t in idx_valid if int(t) not in scores]
    if remaining:
        # curvature of the position path, sampled at most as often as needed
        step = max(1, len(remaining) // max(1, max_keyframes))
        for t in remaining[::step]:
            scores.setdefault(t, float(np.linalg.norm(pos[min(t + 1, n - 1)] - pos[max(t - 1, 0)])))

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    picked = sorted(t for t, _ in ordered[: max(1, max_keyframes)])
    return picked


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def candidate_offsets(cfg: BaseSearchConfig, reach_m: float) -> list[tuple[float, float, float]]:
    """``(lateral, forward, vertical)`` offsets in metres, scaled by reach."""
    out: list[tuple[float, float, float]] = []
    for lat in cfg.lateral_levels:
        for fwd in cfg.forward_levels:
            for vert in cfg.vertical_levels:
                out.append(
                    (
                        cfg.lateral_sign * lat * reach_m,
                        fwd * reach_m,
                        vert * reach_m,
                    )
                )
    return out


def candidate_orientations(cfg: BaseSearchConfig) -> list[tuple[float, float, float]]:
    """``(pitch, yaw, roll)`` triples in degrees."""
    return [(p, y, r) for p in cfg.pitch_deg for y in cfg.yaw_deg for r in cfg.roll_deg]


def _orientation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> NDArray[np.float64]:
    """``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` — camera-frame base tilt."""
    p, y, r = np.radians([pitch_deg, yaw_deg, roll_deg])
    Rx = exp_so3(np.array([r, 0.0, 0.0]))
    Ry = exp_so3(np.array([0.0, p, 0.0]))
    Rz = exp_so3(np.array([0.0, 0.0, y]))
    return Rz @ Ry @ Rx


def _make_candidate_base(
    centroid_C: NDArray[np.float64],
    offset: tuple[float, float, float],
    orientation_deg: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Build ``T_C_B`` from a camera-frame offset and a base orientation."""
    t = centroid_C + np.asarray(offset, dtype=np.float64)
    R = _orientation_matrix(*orientation_deg)
    return make_transform(R, t)


def score_candidates(
    candidates: Iterable[BaseCandidate],
    cfg: BaseSearchConfig,
) -> list[BaseCandidate]:
    """Apply ``S = FR - w * |mean_reach_ratio - target|`` and rank."""
    ranked = list(candidates)
    for c in ranked:
        if c.rejected_reason is not None:
            c.score = float("-inf")
            continue
        c.score = c.feasibility_rate - cfg.reach_penalty_weight * abs(
            c.mean_reach_ratio - cfg.reach_ratio_target
        )
    # deterministic tie-break: score desc, then candidate index asc
    ranked.sort(key=lambda c: (-c.score, c.index))
    return ranked


def search_base_placement(
    robot: Any,
    T_C_TCP: NDArray[np.floating[Any]],
    valid: NDArray[np.bool_] | None,
    *,
    cfg: BaseSearchConfig | None = None,
    log: Any = None,
) -> BaseSearchResult:
    """Run the paper's grid search over base placements.

    Keyframe IK is used for ranking only; the caller must still validate the
    winner over the whole trajectory (implementation-plan.md M4 forbids
    declaring success from sparse keyframes).
    """
    cfg = cfg or BaseSearchConfig()
    T = np.asarray(T_C_TCP, dtype=np.float64)
    reach = float(robot.max_reach_m)
    mask = np.ones(T.shape[0], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if not mask.any():
        raise ValueError("no valid frames in the trajectory; cannot place a base")

    keyframes = select_keyframes(T, mask, max_keyframes=cfg.max_keyframes)
    if not keyframes:
        raise ValueError("keyframe selection returned nothing")
    targets_C = T[keyframes, :3, 3]
    centroid_C = targets_C.mean(axis=0)

    offsets = candidate_offsets(cfg, reach)
    orientations = candidate_orientations(cfg)
    if log is not None:
        log.info(
            f"base search: {len(keyframes)} keyframes, {len(offsets)} offsets x "
            f"{len(orientations)} orientations = {len(offsets) * len(orientations)} candidates"
        )

    robots_cfg = robot.config
    T_E_TCP = robots_cfg.T_E_TCP_array
    home_gripper = np.asarray(robots_cfg.home_gripper_qpos, dtype=np.float64)

    candidates: list[BaseCandidate] = []
    rejected_camera = 0
    rejected_depth = 0
    rejected_reach = 0
    index = 0
    for offset in offsets:
        for orientation in orientations:
            T_C_B = _make_candidate_base(centroid_C, offset, orientation)
            base_pos_C = T_C_B[:3, 3]
            T_B_C = invert_transform(T_C_B)

            camera_distance = float(np.linalg.norm(base_pos_C))
            if camera_distance < cfg.min_camera_distance_m:
                rejected_camera += 1
                candidates.append(
                    BaseCandidate(
                        index=index,
                        T_C_B=T_C_B,
                        offset_reach_units=tuple(v / reach for v in offset),
                        orientation_deg=orientation,
                        rejected_reason="camera_distance",
                    )
                )
                index += 1
                continue

            if float(base_pos_C[2]) < cfg.min_forward_depth_m:
                rejected_depth += 1
                candidates.append(
                    BaseCandidate(
                        index=index,
                        T_C_B=T_C_B,
                        offset_reach_units=tuple(v / reach for v in offset),
                        orientation_deg=orientation,
                        rejected_reason="camera_depth",
                    )
                )
                index += 1
                continue

            targets_B = transform_points(T_B_C, targets_C)
            distances = np.linalg.norm(targets_B, axis=1)
            too_far = bool(np.any(distances > cfg.max_reach_fraction * reach))
            too_close = bool(np.any(distances < cfg.min_base_distance_m))
            if too_far or too_close:
                rejected_reach += 1
                candidates.append(
                    BaseCandidate(
                        index=index,
                        T_C_B=T_C_B,
                        offset_reach_units=tuple(v / reach for v in offset),
                        orientation_deg=orientation,
                        rejected_reason="reach" if too_far else "base_distance",
                    )
                )
                index += 1
                continue

            successes = 0
            pos_errors: list[float] = []
            rot_errors: list[float] = []
            q_init = np.asarray(robots_cfg.home_qpos, dtype=np.float64)
            for k, idx in enumerate(keyframes):
                T_B_TCP = T_B_C @ T[idx]
                res = robot.solve_ik(T_B_TCP, q_init, gripper_qpos=home_gripper)
                if res.success:
                    successes += 1
                    q_init = res.qpos  # warm start from the previous keyframe
                pos_errors.append(res.position_error_m)
                rot_errors.append(res.rotation_error_rad)

            mean_reach = float(np.mean(distances) / reach)
            candidates.append(
                BaseCandidate(
                    index=index,
                    T_C_B=T_C_B,
                    offset_reach_units=tuple(v / reach for v in offset),
                    orientation_deg=orientation,
                    feasibility_rate=successes / float(len(keyframes)),
                    mean_reach_ratio=mean_reach,
                    ik_successes=successes,
                    keyframes_total=len(keyframes),
                    mean_position_error_m=float(np.mean(pos_errors)),
                    mean_rotation_error_rad=float(np.mean(rot_errors)),
                )
            )
            index += 1

    ranked = score_candidates(candidates, cfg)
    feasible = [c for c in ranked if c.rejected_reason is None and c.ik_successes > 0]
    if not feasible:
        raise RuntimeError(
            "base search found no candidate with a single feasible keyframe; the "
            "trajectory may be out of reach for this morphology"
        )
    best = feasible[0]
    notes = [
        "ranking uses keyframe IK only; full-trajectory validation is mandatory "
        "before this placement is accepted",
        f"reach penalty target {cfg.reach_ratio_target}",
    ]
    return BaseSearchResult(
        T_C_B=best.T_C_B,
        best=best,
        ranked=ranked,
        total_candidates=len(candidates),
        rejected_by_camera_distance=rejected_camera,
        rejected_by_reach=rejected_reach,
        keyframe_indices=keyframes,
        config=cfg,
        notes=notes,
        rejected_by_camera_depth=rejected_depth,
    )


_ = compose_transforms
