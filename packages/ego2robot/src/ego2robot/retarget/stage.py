"""``retarget`` stage: HandSequence -> outliers -> gaps -> smoothing -> 7D action.

Implements implementation-plan.md M2.  Produces three comparable trajectories so
the effect of each step is auditable:

``canonical/actions-raw.npz``
    direct output of :func:`~ego2robot.retarget.hand_to_gripper.retarget_hand_sequence`
``canonical/actions-filtered.npz``
    after the two-round velocity filter and gap handling
``canonical/actions.npz``
    the smoothed trajectory — **the only one downstream stages may consume**
``canonical/actions-delta.npz``
    camera-frame relative 7D action derived from the absolutes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core import artifacts as A
from ..core.frames import Frame, transform_name
from ..core.schema import CanonicalActionSequence, HandSequence, ValidationResult
from ..core.transforms import quat_xyzw_from_matrix, so3_repair
from ..pipeline.registry import register_stage
from ..pipeline.stage import StageContext, StageResult
from .gaps import GapConfigLike, fill_gaps
from .hand_to_gripper import (
    MANO_INDEX_TIP,
    MANO_MIDDLE_TIP,
    MANO_THUMB_TIP,
    MANO_WRIST,
    RetargetConfigLike,
    retarget_hand_sequence,
)
from .outliers import OutlierConfigLike, reject_outliers
from .smoothing import SmoothingConfigLike, smooth_trajectory

__all__ = ["RetargetStage", "build_canonical_sequence"]


def _config_likes(cfg: Any) -> tuple[RetargetConfigLike, OutlierConfigLike, GapConfigLike, SmoothingConfigLike]:
    retarget = RetargetConfigLike(
        virtual_fingertip_index_weight=cfg.retarget.virtual_fingertip_index_weight,
        virtual_fingertip_middle_weight=cfg.retarget.virtual_fingertip_middle_weight,
        handedness_right_sign=cfg.retarget.handedness_right_sign,
        handedness_left_sign=cfg.retarget.handedness_left_sign,
        min_gripper_width_m=cfg.retarget.degeneracy.min_gripper_width_m,
        parallel_dot_threshold=cfg.retarget.degeneracy.parallel_dot_threshold,
        repair_so3=cfg.retarget.repair_so3,
        max_so3_correction_rad=cfg.retarget.max_so3_correction_rad,
    )
    outliers = OutlierConfigLike(
        enabled=cfg.outliers.enabled,
        position_median_multiplier=cfg.outliers.position_median_multiplier,
        rotation_median_multiplier=cfg.outliers.position_median_multiplier,
        position_floor_per_fps_m=cfg.outliers.position_floor_per_fps_m,
        rotation_floor_per_fps_rad=cfg.outliers.rotation_floor_per_fps_rad,
        rounds=cfg.outliers.rounds,
        min_samples_for_median=cfg.outliers.min_samples_for_median,
    )
    gaps = GapConfigLike(
        small_gap_max_frames=cfg.gaps.small_gap_max_frames,
        blend_min_frames=cfg.gaps.blend_min_frames,
        blend_max_frames=cfg.gaps.blend_max_frames,
        blend_position_speed_m_per_frame=cfg.gaps.blend_position_speed_m_per_frame,
        blend_rotation_speed_rad_per_frame=cfg.gaps.blend_rotation_speed_rad_per_frame,
        blend_position_weight=cfg.gaps.blend_position_weight,
        blend_rotation_weight=cfg.gaps.blend_rotation_weight,
        home_gripper_width_m=cfg.gaps.home_gripper_width_m,
        home_source=cfg.gaps.home_source,
    )
    smoothing = SmoothingConfigLike(
        position_window=cfg.smoothing.position_window,
        position_polyorder=cfg.smoothing.position_polyorder,
        width_window=cfg.smoothing.width_window,
        width_polyorder=cfg.smoothing.width_polyorder,
        orientation_sigma_frames=cfg.smoothing.orientation_sigma_frames,
        orientation_kernel_size=cfg.smoothing.orientation_kernel_size,
        enforce_odd_window=cfg.smoothing.enforce_odd_window,
    )
    return retarget, outliers, gaps, smoothing


def build_canonical_sequence(
    source: CanonicalActionSequence,
    *,
    position_C_m: np.ndarray,
    rotation_C_TCP: np.ndarray,
    width_m: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    fallback_code: np.ndarray,
    so3_correction_rad: np.ndarray,
    stage_tag: str,
    extra_meta: dict[str, Any] | None = None,
) -> CanonicalActionSequence:
    """Assemble a persisted canonical sequence from smoothed/gap-filled arrays.

    Every array carries the ``[T, H, ...]`` hand axis (ADR-0021): the per-hand
    retarget, filtering, gap filling and smoothing all run independently and are
    stacked along ``H`` here, in the adapter's ``hand_ids`` order.
    """
    position_C_m = np.asarray(position_C_m, dtype=np.float64)
    rotation_C_TCP = np.asarray(rotation_C_TCP, dtype=np.float64)
    if position_C_m.ndim != 3 or position_C_m.shape[2] != 3:
        raise ValueError(f"position_C_m must be [T, H, 3], got {position_C_m.shape}")
    T, H = position_C_m.shape[:2]
    if rotation_C_TCP.shape[:2] != (T, H):
        raise ValueError(
            f"rotation_C_TCP must be [T, H, 3, 3] with T={T}, H={H}, got {rotation_C_TCP.shape}"
        )
    quats = np.full((T, H, 4), np.nan, dtype=np.float64)
    transforms = np.full((T, H, 4, 4), np.nan, dtype=np.float64)
    for h in range(H):
        for t in range(T):
            R = rotation_C_TCP[t, h]
            if np.isfinite(R).all():
                quats[t, h] = quat_xyzw_from_matrix(R)
                transforms[t, h, :3, :3] = R
                transforms[t, h, :3, 3] = position_C_m[t, h]
                # the homogeneous bottom row must be written explicitly: leaving it
                # as NaN produces a matrix that silently poisons every downstream
                # matrix product while looking "mostly right" in a dump
                transforms[t, h, 3, :] = (0.0, 0.0, 0.0, 1.0)
    meta = {
        "stage_tag": stage_tag,
        "pose_variable": str(transform_name(Frame.C, Frame.TCP)),
        "axes": {"x": "approach", "y": "gripper normal (jaw-plane normal)", "z": "grasp axis"},
        "hand_ids": list(source.hand_ids),
    }
    meta.update(extra_meta or {})
    return CanonicalActionSequence(
        {
            "timestamp_s": np.asarray(source["timestamp_s"], dtype=np.float64),
            "frame_index": np.asarray(source["frame_index"], dtype=np.int64),
            "T_C_TCP": transforms,
            "tcp_position_C_m": position_C_m,
            "tcp_quaternion_C_xyzw": quats,
            "gripper_width_m": np.asarray(width_m, dtype=np.float64),
            "confidence": np.asarray(confidence, dtype=np.float32),
            "valid": np.asarray(valid, dtype=bool),
            "fallback_code": np.asarray(fallback_code, dtype=np.uint16),
            "so3_correction_rad": np.asarray(so3_correction_rad, dtype=np.float32),
        },
        meta=meta,
    )


@register_stage
class RetargetStage:
    """``retarget``: the perception/robot boundary (system-design.md 6.3)."""

    name = "retarget"
    version = "2"

    # -- contract ----------------------------------------------------------

    def _load_hand(self, ctx: StageContext) -> HandSequence:
        if not ctx.layout.hand_npz.is_file():
            raise FileNotFoundError(
                f"{ctx.layout.hand_npz} is missing; run the 'hand' stage first"
            )
        return HandSequence.load_npz(ctx.layout.hand_npz)

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        res = ValidationResult()
        try:
            seq = self._load_hand(ctx)
        except FileNotFoundError as exc:
            res.add("retarget.no_hand_sequence", str(exc))
            return res
        check = seq.validate()
        res.extend(check)
        res.extend(seq.validate_hand_identity())
        if res.ok:
            kp = np.asarray(seq["keypoints_C_m"])
            required_indices = (MANO_WRIST, MANO_THUMB_TIP, MANO_INDEX_TIP, MANO_MIDDLE_TIP)
            finite_required = np.isfinite(kp[:, :, required_indices, :]).all(axis=(2, 3))
            valid = np.asarray(seq["valid"], dtype=np.bool_)
            hand_ids = seq.hand_ids or [f"hand{i}" for i in range(seq.num_hands)]
            for h, hand_id in enumerate(hand_ids):
                missing = int(np.count_nonzero(valid[:, h] & ~finite_required[:, h]))
                if missing:
                    res.add(
                        "retarget.nan_in_valid_frames",
                        f"hand {hand_id!r}: {missing} valid hand-frame(s) lack one or more "
                        "required retarget keypoints (wrist/thumb/index/middle tip); the "
                        "retargeter cannot produce a TCP frame from them",
                    )
        return res

    def fingerprint(self, ctx: StageContext) -> str:
        hand_manifest = A.read_stage_manifest(ctx.layout.manifest_path("hand"))
        return A.hash_json(
            {
                "hand_fingerprint": hand_manifest.input_fingerprints.get("self", "") if hand_manifest else "",
                "hand_npz": A.file_sha256(ctx.layout.hand_npz) if ctx.layout.hand_npz.is_file() else "",
                "retarget": ctx.config.retarget.model_dump(mode="json"),
                "outliers": ctx.config.outliers.model_dump(mode="json"),
                "gaps": ctx.config.gaps.model_dump(mode="json"),
                "smoothing": ctx.config.smoothing.model_dump(mode="json"),
                "stage_version": self.version,
            }
        )

    def run(self, ctx: StageContext) -> StageResult:
        result = StageResult(stage=self.name)
        cfg = ctx.config
        retarget_cfg, outlier_cfg, gap_cfg, smooth_cfg = _config_likes(cfg)
        sequence = self._load_hand(ctx)

        episode = (
            A.read_json(ctx.layout.episode_json) if ctx.layout.episode_json.is_file() else {}
        )
        fps = float(episode.get("timebase", {}).get("nominal_fps", 30.0)) or 30.0

        raw_results: list[Any] = []
        filtered_results: list[Any] = []
        gap_results: list[Any] = []
        smoothed_results: list[Any] = []
        repaired_rotations: list[np.ndarray] = []

        hand_ids = sequence.hand_ids or [f"hand{i}" for i in range(sequence.num_hands)]
        n_hands = sequence.num_hands
        for h in range(n_hands):
            # ADR-0021: every temporal step runs per hand, so one hand's dropouts
            # cannot shift the other hand's thresholds or smoothing windows.
            raw_result, _unused = retarget_hand_sequence(
                sequence, hand_index=h, cfg=retarget_cfg
            )
            filtered = reject_outliers(
                raw_result.position_C_m,
                raw_result.rotation_C_TCP,
                raw_result.gripper_width_m,
                raw_result.valid,
                fps=fps,
                cfg=outlier_cfg,
            )
            gap_result = fill_gaps(
                filtered.position_C_m,
                filtered.rotation_C_TCP,
                filtered.width_m,
                raw_result.valid,
                cfg=gap_cfg,
            )
            hand_valid = np.asarray(raw_result.valid, dtype=bool).copy()
            hand_valid |= gap_result.filled
            smoothed = smooth_trajectory(
                gap_result.position_C_m,
                gap_result.rotation_C_TCP,
                gap_result.width_m,
                valid=hand_valid,
                cfg=smooth_cfg,
            )
            repaired = np.stack([so3_repair(R).R for R in smoothed.rotation_C_TCP])
            raw_results.append(raw_result)
            filtered_results.append(filtered)
            gap_results.append(gap_result)
            smoothed_results.append(smoothed)
            repaired_rotations.append(repaired)

        def _stack(results: list[Any], attribute: str) -> np.ndarray:
            return np.stack([np.asarray(getattr(r, attribute)) for r in results], axis=1)

        # --- raw ------------------------------------------------------------
        raw_action = build_canonical_sequence(
            sequence,
            position_C_m=_stack(raw_results, "position_C_m"),
            rotation_C_TCP=_stack(raw_results, "rotation_C_TCP"),
            width_m=_stack(raw_results, "gripper_width_m"),
            confidence=_stack(raw_results, "confidence"),
            valid=_stack(raw_results, "valid"),
            fallback_code=_stack(raw_results, "fallback_code"),
            so3_correction_rad=_stack(raw_results, "so3_correction_rad"),
            stage_tag="raw",
            extra_meta={
                "raw_output": True,
                "handedness": {
                    hand_id: int(np.asarray(sequence["handedness"])[0, h]) if sequence.num_frames else 0
                    for h, hand_id in enumerate(hand_ids)
                },
            },
        )
        result.add_output(raw_action.save_npz(ctx.layout.raw_actions_npz))

        # --- velocity filter + gaps ----------------------------------------
        filtered_valid = np.stack(
            [
                np.asarray(raw.valid, dtype=bool) | gap.filled
                for raw, gap in zip(raw_results, gap_results)
            ],
            axis=1,
        )
        one_hand = n_hands == 1
        filtered_action = build_canonical_sequence(
            sequence,
            position_C_m=_stack(gap_results, "position_C_m"),
            rotation_C_TCP=_stack(gap_results, "rotation_C_TCP"),
            width_m=_stack(gap_results, "width_m"),
            confidence=_stack(raw_results, "confidence"),
            valid=filtered_valid,
            fallback_code=_stack(raw_results, "fallback_code"),
            so3_correction_rad=_stack(raw_results, "so3_correction_rad"),
            stage_tag="filtered",
            extra_meta={
                "outlier_rejected_frames": _per_hand(
                    hand_ids, [f.rejected_count for f in filtered_results], one_hand
                ),
                "gap_spans": _per_hand(
                    hand_ids, [[list(s) for s in g.gap_spans] for g in gap_results], one_hand
                ),
            },
        )
        result.add_output(filtered_action.save_npz(ctx.layout.filtered_actions_npz))

        # --- smoothing ------------------------------------------------------
        repaired_rotation = np.stack(repaired_rotations, axis=1)
        final_action = build_canonical_sequence(
            sequence,
            position_C_m=_stack(smoothed_results, "position_C_m"),
            rotation_C_TCP=repaired_rotation,
            width_m=_stack(smoothed_results, "width_m"),
            confidence=_stack(raw_results, "confidence"),
            valid=filtered_valid,
            fallback_code=_stack(raw_results, "fallback_code"),
            so3_correction_rad=_stack(raw_results, "so3_correction_rad"),
            stage_tag="smoothed",
            extra_meta={
                "smoothing": _per_hand(
                    hand_ids, [s.applied for s in smoothed_results], one_hand
                )
            },
        )
        final_check = final_action.validate()
        final_check.raise_if_bad()
        result.add_output(final_action.save_npz(ctx.layout.actions_npz))

        # --- camera-frame relative 7D action --------------------------------
        deltas = final_action.relative_action()
        delta_payload = {k: np.asarray(v) for k, v in deltas.items()}
        np.savez_compressed(ctx.layout.delta_npz, **delta_payload)
        result.add_output(ctx.layout.delta_npz)

        # --- per-hand previews (AGENTS.md rule 8) ---------------------------
        if not ctx.dry_run:
            from .preview import write_retarget_previews

            try:
                previews = write_retarget_previews(
                    action=final_action,
                    raw_action=raw_action,
                    previews_dir=ctx.layout.previews_dir,
                    episode_id=ctx.episode_id,
                )
                for p in previews:
                    result.add_output(p)
            except Exception as exc:  # noqa: BLE001 - previews must not invalidate data
                result.warnings.append(
                    f"retarget preview generation failed ({type(exc).__name__}: {exc}); the "
                    "canonical arrays themselves are still valid"
                )

        # --- metrics --------------------------------------------------------
        valid = np.asarray(final_action["valid"], dtype=bool)
        widths_all = np.asarray(final_action["gripper_width_m"], dtype=np.float64)
        fallback_all = np.asarray(final_action["fallback_code"])
        so3_all = np.asarray(final_action["so3_correction_rad"], dtype=np.float64)
        pos_steps = np.linalg.norm(delta_payload["delta_position_C_m"], axis=2)
        rot_steps = np.linalg.norm(delta_payload["delta_rotvec_C_rad"], axis=2)

        per_hand: dict[str, Any] = {}
        for h, hand_id in enumerate(hand_ids):
            column = valid[:, h]
            widths = widths_all[:, h]
            finite = np.isfinite(widths) & column
            per_hand[hand_id] = {
                "valid_frames": int(np.count_nonzero(column)),
                "valid_ratio": round(float(np.mean(column)), 6) if column.size else 0.0,
                "fallback_frames": int(np.count_nonzero(fallback_all[:, h] != 0)),
                "frozen_frames": raw_results[h].diagnostics["frozen_frames"],
                "degenerate_frames": raw_results[h].diagnostics["degenerate_frames"],
                "so3_max_correction_rad": round(float(np.max(so3_all[:, h])), 9) if column.size else 0.0,
                "outlier_rejected_frames": filtered_results[h].rejected_count,
                "outlier_position_threshold_m": round(filtered_results[h].position_threshold_m, 9),
                "outlier_rotation_threshold_rad": round(filtered_results[h].rotation_threshold_rad, 9),
                "outlier_per_round": filtered_results[h].per_round_rejected,
                "small_gaps": gap_results[h].small_gaps,
                "large_gaps": gap_results[h].large_gaps,
                "gap_spans": [list(s) for s in gap_results[h].gap_spans],
                "blend_lengths": gap_results[h].blend_lengths,
                "home_was_derived": gap_results[h].home_was_derived,
                "smoothing": smoothed_results[h].applied,
                "gripper_width_min_m": round(float(np.min(widths[finite])), 6) if finite.any() else None,
                "gripper_width_median_m": round(float(np.median(widths[finite])), 6) if finite.any() else None,
                "gripper_width_max_m": round(float(np.max(widths[finite])), 6) if finite.any() else None,
                "delta_position_step_median_m": round(float(np.median(pos_steps[:, h])), 9),
                "delta_position_step_max_m": round(float(np.max(pos_steps[:, h])), 9),
                "delta_rotvec_step_median_rad": round(float(np.median(rot_steps[:, h])), 9),
                "delta_rotvec_step_max_rad": round(float(np.max(rot_steps[:, h])), 9),
            }
        first = per_hand[hand_ids[0]]
        jointly_valid = np.all(valid, axis=1) if valid.ndim == 2 else valid
        result.metrics.update(
            {
                "num_frames": final_action.num_frames,
                "num_hands": n_hands,
                "hand_ids": list(hand_ids),
                # a frame counts as valid only when every hand is valid; the
                # per-hand numbers are in `per_hand` so one bad hand cannot be
                # hidden by a good one
                "valid_frames": int(np.count_nonzero(jointly_valid)),
                "valid_ratio": round(float(np.mean(jointly_valid)), 6) if jointly_valid.size else 0.0,
                "valid_hand_frames": int(np.count_nonzero(valid)),
                "jointly_valid_frames": int(np.count_nonzero(jointly_valid)),
                "jointly_valid_ratio": round(float(np.mean(jointly_valid)), 6) if jointly_valid.size else 0.0,
                "fallback_frames": int(np.count_nonzero(fallback_all != 0)),
                "frozen_frames": sum(int(r.diagnostics["frozen_frames"]) for r in raw_results),
                "degenerate_frames": sum(int(r.diagnostics["degenerate_frames"]) for r in raw_results),
                "so3_max_correction_rad": round(float(np.max(so3_all)), 9) if so3_all.size else 0.0,
                "outlier_rejected_frames": sum(int(f.rejected_count) for f in filtered_results),
                "small_gaps": sum(int(g.small_gaps) for g in gap_results),
                "large_gaps": sum(int(g.large_gaps) for g in gap_results),
                "gripper_width_min_m": first.get("gripper_width_min_m"),
                "gripper_width_median_m": first.get("gripper_width_median_m"),
                "gripper_width_max_m": first.get("gripper_width_max_m"),
                "delta_position_step_median_m": round(float(np.median(pos_steps)), 9),
                "delta_position_step_max_m": round(float(np.max(pos_steps)), 9),
                "delta_rotvec_step_median_rad": round(float(np.median(rot_steps)), 9),
                "delta_rotvec_step_max_rad": round(float(np.max(rot_steps)), 9),
                # single-hand runs keep the flat scalar fields the old manifests had
                "outlier_position_threshold_m": first.get("outlier_position_threshold_m"),
                "outlier_rotation_threshold_rad": first.get("outlier_rotation_threshold_rad"),
                "outlier_per_round": first.get("outlier_per_round"),
                "gap_spans": first.get("gap_spans"),
                "blend_lengths": first.get("blend_lengths"),
                "home_was_derived": first.get("home_was_derived"),
                "smoothing": first.get("smoothing"),
                "per_hand": per_hand,
            }
        )

        for revision in ("raw", "filtered", "smoothed"):
            _validate_rotations(
                {
                    "raw": _stack(raw_results, "rotation_C_TCP"),
                    "filtered": _stack(gap_results, "rotation_C_TCP"),
                    "smoothed": repaired_rotation,
                }[revision],
                valid_mask=valid if revision == "smoothed" else _stack(raw_results, "valid"),
                revision=revision,
                result=result,
                hand_ids=hand_ids,
            )

        result.payload["action"] = final_action
        result.payload["deltas"] = delta_payload
        result.payload["hand_sequence"] = sequence
        result.payload["episode"] = episode
        return result

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        res = ValidationResult()
        action = result.payload.get("action")
        if not isinstance(action, CanonicalActionSequence):
            res.add("retarget.no_action", "stage produced no CanonicalActionSequence payload")
            return res
        res.extend(action.validate())
        res.extend(action.validate_hand_identity())
        res.extend(
            action.validate_numeric(
                allow_nan_in=("T_C_TCP", "tcp_position_C_m", "tcp_quaternion_C_xyzw", "gripper_width_m")
            )
        )
        if res.ok:
            hand_ids = action.hand_ids or [f"hand{i}" for i in range(action.num_hands)]
            valid = np.asarray(action["valid"], dtype=bool)
            transforms = np.asarray(action["T_C_TCP"], dtype=np.float64)
            for h, hand_id in enumerate(hand_ids):
                R_all = transforms[:, h, :3, :3]
                for t in np.nonzero(valid[:, h])[0]:
                    R = R_all[t]
                    RtR = R.T @ R
                    if not np.allclose(RtR, np.eye(3), atol=1e-6):
                        res.add(
                            "retarget.not_orthonormal",
                            f"hand {hand_id!r} frame {t}: R^T R deviates from I by "
                            f"{np.max(np.abs(RtR - np.eye(3))):.2e}",
                        )
                        break
                    if abs(float(np.linalg.det(R)) - 1.0) > 1e-6:
                        res.add(
                            "retarget.determinant",
                            f"hand {hand_id!r} frame {t}: det(R) = {np.linalg.det(R):.6f}",
                        )
                        break
        for path in result.outputs:
            if not Path(path).is_file():
                res.add("retarget.missing_output", f"declared output does not exist: {path}")
        return res


def _validate_rotations(
    rotation: np.ndarray,
    *,
    valid_mask: np.ndarray,
    revision: str,
    result: StageResult,
    hand_ids: list[str] | None = None,
) -> None:
    """Every valid rotation must satisfy R^T R ~= I and det R ~= 1 (Gate M2).

    ``rotation`` is ``[T, H, 3, 3]`` and ``valid_mask`` ``[T, H]`` (ADR-0021);
    both are checked per hand so a good hand cannot mask a broken one.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    rotation = np.asarray(rotation, dtype=np.float64)
    n_hands = rotation.shape[1]
    ids = hand_ids or [f"hand{h}" for h in range(n_hands)]
    worst_ortho = 0.0
    worst_det = 0.0
    per_hand: dict[str, dict[str, float]] = {}
    for h in range(n_hands):
        hand_ortho = 0.0
        hand_det = 0.0
        for t in np.nonzero(valid_mask[:, h])[0]:
            R = rotation[t, h]
            if not np.isfinite(R).all():
                result.warnings.append(
                    f"{revision}: hand {ids[h]} frame {t} is valid but its rotation is not finite"
                )
                continue
            hand_ortho = max(hand_ortho, float(np.max(np.abs(R.T @ R - np.eye(3)))))
            hand_det = max(hand_det, abs(float(np.linalg.det(R)) - 1.0))
        per_hand[ids[h]] = {
            "max_orthogonality_error": round(hand_ortho, 12),
            "max_determinant_error": round(hand_det, 12),
        }
        worst_ortho = max(worst_ortho, hand_ortho)
        worst_det = max(worst_det, hand_det)
    result.metrics[f"{revision}_max_orthogonality_error"] = round(worst_ortho, 12)
    result.metrics[f"{revision}_max_determinant_error"] = round(worst_det, 12)
    if n_hands > 1:
        result.metrics[f"{revision}_rotation_check_per_hand"] = per_hand


def _per_hand(hand_ids: list[str], values: list[Any], single: bool) -> Any:
    """Keep the old scalar metadata for one-hand runs, a mapping for two hands."""
    return values[0] if single and values else {hid: v for hid, v in zip(hand_ids, values)}
