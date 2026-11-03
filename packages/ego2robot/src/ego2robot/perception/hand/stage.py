"""Hand stage: canonical :class:`HandSequence` from Path A annotations.

Implements implementation-plan.md M1.  The stage is deliberately thin: all
dataset knowledge lives in the adapters (:mod:`ego2robot.ingest.egodex`,
:mod:`ego2robot.ingest.fixture`), and this stage only binds them to the
pipeline contract and validates the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...core import artifacts as A
from ...core.schema import CameraModel, EpisodeManifest, HandSequence, ValidationResult
from ...ingest.egodex import EgoDexHandAdapter, is_egodex_hdf5, probe_egodex_hdf5
from ...ingest.fixture import HandAnnotations, list_hands, read_hand_annotations
from ...pipeline.registry import register_stage
from ...pipeline.stage import StageContext, StageResult
from .base import HandPoseProvider

__all__ = ["HandStage", "FixtureHandProvider"]


class FixtureHandProvider:
    """Path A provider over this project's own fixture HDF5 layout.

    See :mod:`ego2robot.ingest.fixture`.  Kept separate from
    :class:`~ego2robot.ingest.egodex.EgoDexHandAdapter` so a bug in one cannot
    mask a bug in the other.
    """

    name = "fixture_hdf5"
    version = "1"

    def __init__(self, hand: str = "right", min_confidence: float = 0.0) -> None:
        self.hand = hand
        self.min_confidence = min_confidence

    def load(
        self,
        source: str | Path,
        *,
        num_frames: int | None = None,
        frame_offset: int = 0,
        camera: CameraModel | None = None,
    ) -> tuple[HandSequence, dict[str, Any]]:
        ann: HandAnnotations = read_hand_annotations(source, self.hand)
        n_total = len(ann)
        n = n_total if num_frames is None else min(num_frames, n_total - frame_offset)
        if n <= 0:
            raise ValueError(f"{source}: requested {num_frames} frames from offset {frame_offset} of {n_total}")
        sl = slice(frame_offset, frame_offset + n)

        keypoints = np.asarray(ann.keypoints_C_m[sl], dtype=np.float32)
        confidence = np.asarray(ann.confidence[sl], dtype=np.float32)
        handedness = np.asarray(ann.handedness[sl], dtype=np.int8)
        if handedness.size and int(handedness[0]) == 0:
            handedness = np.full_like(handedness, -1 if self.hand == "left" else 1)

        if camera is not None:
            uv = camera.project(keypoints.astype(np.float64)).astype(np.float32)
        else:
            uv = np.full((n, 21, 2), np.nan, dtype=np.float32)

        kp_conf = (
            np.asarray(ann.keypoint_confidence[sl], dtype=np.float32)
            if ann.keypoint_confidence is not None
            else np.ones((n, 21), dtype=np.float32)
        )

        valid = (confidence >= self.min_confidence) & np.isfinite(keypoints[:, 0, :]).all(axis=1)
        seq = HandSequence(
            {
                "timestamp_s": np.arange(n, dtype=np.float64) / 30.0,
                "frame_index": np.arange(n, dtype=np.int64) + frame_offset,
                "keypoints_C_m": keypoints[:, None, :, :],
                "keypoints_uv_px": uv[:, None, :, :],
                "handedness": handedness.reshape(n, 1),
                "detection_confidence": confidence.reshape(n, 1),
                "valid": valid.reshape(n, 1),
                "keypoint_confidence": kp_conf[:, None, :],
            },
            meta={
                "provider": self.name,
                "provider_version": self.version,
                "source_hdf5": str(source),
                "hand": self.hand,
                "source_joint_order": "mano21",
                "target_joint_order": "mano21",
            },
        )
        provenance = {
            "provider": self.name,
            "provider_version": self.version,
            "source_hdf5": str(source),
            "source_frame": "C",
            "target_frame": "C",
            "length_unit_in": "m",
            "length_unit_out": "m",
            "unit_conversion": "none (fixture is already metric in the camera frame)",
            "source_joint_order": "mano21",
            "target_joint_order": "mano21",
            "handedness": int(handedness[0]) if handedness.size else 0,
            "num_frames_total": n_total,
            "num_frames_used": n,
            "confidence_source": "fixture detection_confidence",
            "annotation_attributes": {
                k: str(v) for k, v in ann.attributes.items() if not isinstance(v, bytes)
            },
        }
        return seq, provenance


@register_stage
class HandStage:
    """``hand``: annotations -> ``hand/sequence.npz`` + provenance + previews."""

    name = "hand"
    version = "2"

    def _find_video(self, ctx: StageContext) -> Path | None:
        from ...ingest.stage import discover_sources

        video, _hdf5, _cam = discover_sources(ctx.layout.root)
        return video

    # -- provider selection ------------------------------------------------

    def _select_provider(self, ctx: StageContext, hdf5: Path) -> tuple[HandPoseProvider, str]:
        """Pick the adapter by *inspecting the file*, not by trusting the config.

        Provider choice is recorded in the provenance so a surprising result can
        always be traced back to which adapter read the annotations.

        ADR-0021: a multi-hand selection reads the requested sides in the
        configured order and fails when a side is absent from the file -- the
        two-hand contract does not allow a silently missing hand.
        """
        cfg = ctx.config.hand
        want = cfg.provider
        if want in {"egodex_hdf5", "auto"} and is_egodex_hdf5(hdf5):
            probe = probe_egodex_hdf5(hdf5)
            if cfg.multi_hand:
                sides = cfg.resolved_sides
                missing = [s for s in sides if f"{s}Hand" not in probe.joint_names]
                if missing:
                    raise ValueError(
                        f"{hdf5}: hand.selection={cfg.selection!r} requests side(s) {sides} "
                        f"but the annotation has no transform(s) for {missing}; the file "
                        f"provides {sorted(j for j in probe.joint_names if j.endswith('Hand'))}"
                    )
                return (
                    EgoDexHandAdapter(
                        hands=tuple(sides), min_confidence=cfg.min_detection_confidence
                    ),
                    "egodex_hdf5:" + "+".join(sides),
                )
            hand = "right" if "rightHand" in probe.joint_names else "left"
            return (
                EgoDexHandAdapter(hand=hand, min_confidence=cfg.min_detection_confidence),
                "egodex_hdf5",
            )
        if want == "egodex_hdf5":
            raise ValueError(
                f"{hdf5}: hand.provider is pinned to 'egodex_hdf5' but the file has no "
                "/transforms/camera group, so the EgoDex adapter cannot read it"
            )
        if cfg.multi_hand:
            raise ValueError(
                f"{hdf5}: hand.selection={cfg.selection!r} needs an adapter that can read "
                "both hands in one pass; the fixture layout stores one hand per file "
                "(ADR-0021)"
            )
        groups = list_hands(hdf5)
        hand = groups[0]
        return (
            FixtureHandProvider(hand=hand, min_confidence=cfg.min_detection_confidence),
            f"fixture_hdf5:{hand}",
        )

    def _find_hdf5(self, ctx: StageContext) -> Path:
        episode = self._episode(ctx)
        if episode.source.hdf5_uri:
            candidate = ctx.layout.root / episode.source.hdf5_uri
            if candidate.is_file():
                return candidate
        for f in sorted(ctx.layout.source_dir.glob("*.hdf5")):
            return f
        raise FileNotFoundError(f"no HDF5 hand annotation under {ctx.layout.source_dir}")

    @staticmethod
    def _episode(ctx: StageContext) -> EpisodeManifest:
        if not ctx.layout.episode_json.is_file():
            raise FileNotFoundError(
                f"{ctx.layout.episode_json} is missing; run the 'ingest' stage first"
            )
        return EpisodeManifest.model_validate(A.read_json(ctx.layout.episode_json))

    # -- contract ----------------------------------------------------------

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        res = ValidationResult()
        try:
            episode = self._episode(ctx)
        except FileNotFoundError as exc:
            res.add("hand.no_episode_manifest", str(exc))
            return res
        if episode.camera.calibration_source == "assumed" and not episode.camera.debug_only:
            res.add(
                "hand.unlabelled_assumed_camera",
                "camera is 'assumed' but not flagged debug_only",
            )
        try:
            self._find_hdf5(ctx)
        except FileNotFoundError as exc:
            res.add("hand.no_annotations", str(exc))
        return res

    def fingerprint(self, ctx: StageContext) -> str:
        hdf5 = self._find_hdf5(ctx)
        ingest_manifest = A.read_stage_manifest(ctx.layout.manifest_path("ingest"))
        ingest_fp = (
            ingest_manifest.input_fingerprints.get("self", "") if ingest_manifest else ""
        )
        return A.hash_json(
            {
                "hdf5": A.file_sha256(hdf5),
                "ingest_fingerprint": ingest_fp,
                "provider": ctx.config.hand.provider,
                "min_confidence": ctx.config.hand.min_detection_confidence,
                "hand_count": ctx.config.hand.hand_count,
                "selection": ctx.config.hand.selection,
                "sides": ctx.config.hand.resolved_sides,
                "stage_version": self.version,
            }
        )

    def run(self, ctx: StageContext) -> StageResult:
        result = StageResult(stage=self.name)
        episode = self._episode(ctx)
        hdf5 = self._find_hdf5(ctx)

        provider, provider_tag = self._select_provider(ctx, hdf5)
        sequence, provenance = provider.load(
            hdf5,
            num_frames=episode.timebase.num_frames,
            camera=episode.camera,
        )

        # Re-derive timestamps from the episode PTS so hand and video share one
        # clock (system-design.md 5.3).  The adapter's uniform grid is only a
        # fallback when the container gave no PTS.
        if episode.timebase.timestamps_s is not None:
            ts = np.asarray(episode.timebase.timestamps_s, dtype=np.float64)
            if ts.size == sequence.num_frames:
                sequence.arrays["timestamp_s"] = ts
            elif ts.size > sequence.num_frames:
                sequence.arrays["timestamp_s"] = ts[: sequence.num_frames]
            else:
                result.warnings.append(
                    f"timebase has {ts.size} timestamps but the hand sequence has "
                    f"{sequence.num_frames} frames; kept the adapter clock"
                )
        sequence.arrays["frame_index"] = np.arange(sequence.num_frames, dtype=np.int64)

        check = sequence.validate(expect_hands=ctx.config.hand.hand_count)
        check.extend(sequence.validate_numeric(allow_nan_in=("keypoints_C_m", "keypoints_uv_px")))
        check.extend(sequence.validate_hand_identity())
        check.raise_if_bad()

        hand_ids = sequence.hand_ids
        valid = np.asarray(sequence["valid"])
        finite_wrist = np.isfinite(np.asarray(sequence["keypoints_C_m"])[:, :, 0, :]).all(axis=2)
        bad_hands = [
            f"{hand_ids[h] if h < len(hand_ids) else h}: {int(np.count_nonzero(valid[:, h] & ~finite_wrist[:, h]))}"
            for h in range(valid.shape[1])
            if np.any(valid[:, h] & ~finite_wrist[:, h])
        ]
        if bad_hands:
            raise ValueError(
                f"valid frames with a NaN wrist in hand(s) {bad_hands}; missing frames must "
                "be flagged invalid, never zero-filled (implementation-plan.md Gate M1)"
            )

        sequence.meta.update(
            {
                "provenance": provenance,
                "provider_tag": provider_tag,
                "source_frame": provenance.get("source_frame", "?"),
                "target_frame": "C",
                "length_unit": "m",
                "keypoint_convention": "mano21",
                "hand_ids": list(hand_ids),
            }
        )
        out_path = sequence.save_npz(ctx.layout.hand_npz)
        result.add_output(out_path)
        result.add_output(A.atomic_write_json(ctx.layout.hand_dir / "provenance.json", provenance))

        # ---- ADR-0021 per-hand validity record ----------------------------
        validity_payload = {
            "schema_version": "ego2robot.hand_validity.v1",
            "episode_id": ctx.episode_id,
            "num_frames": sequence.num_frames,
            "num_hands": sequence.num_hands,
            "hand_ids": list(hand_ids),
            "selection": ctx.config.hand.selection,
            "confidence_source": provenance.get("confidence_source", "unknown"),
            "notes": [
                "valid is per hand and comes from the adapter: confidence threshold, finite "
                "wrist and (when configured) a depth range",
                "a static hand can still be valid; valid=false means the annotation was "
                "missing/low-confidence, not that the hand did not move",
            ],
            "per_hand": {},
        }
        for h, hand_id in enumerate(hand_ids or [f"hand{i}" for i in range(sequence.num_hands)]):
            column = np.asarray(valid[:, h], dtype=bool)
            validity_payload["per_hand"][hand_id] = {
                "valid_frames": int(np.count_nonzero(column)),
                "invalid_frames": int(np.count_nonzero(~column)),
                "valid_ratio": round(float(np.mean(column)), 6) if column.size else 0.0,
                "first_invalid_frame": (
                    int(np.nonzero(~column)[0][0]) if np.any(~column) else None
                ),
                "frames": [int(v) for v in np.nonzero(~column)[0]],
            }
        result.add_output(
            A.atomic_write_json(ctx.layout.hand_dir / "validity-per-hand.json", validity_payload)
        )

        # ---- M1 evidence artifacts (AGENTS.md rule 8) ----------------------
        video = self._find_video(ctx)
        if video is not None:
            from .preview import write_hand_previews

            try:
                previews = write_hand_previews(
                    sequence=sequence,
                    camera=episode.camera,
                    video_path=video,
                    previews_dir=ctx.layout.previews_dir,
                    hand_ids=hand_ids or None,
                    case_id=ctx.episode_id,
                    num_stills=6 if sequence.num_hands > 1 else 4,
                    dry_run=ctx.dry_run,
                )
                for p in previews:
                    result.add_output(p)
                result.metrics["preview_artifacts"] = len(previews)
            except Exception as exc:  # noqa: BLE001 - previews must not invalidate data
                result.warnings.append(
                    f"hand preview generation failed ({type(exc).__name__}: {exc}); the "
                    "hand sequence itself is still valid"
                )
        else:
            result.warnings.append("no video found; hand overlay previews were skipped")

        keypoints_C = np.asarray(sequence["keypoints_C_m"])
        per_hand: dict[str, Any] = {}
        for h, hand_id in enumerate(hand_ids or [f"hand{i}" for i in range(sequence.num_hands)]):
            depth = keypoints_C[:, h, 0, 2]
            depth_valid = depth[valid[:, h]]
            column = valid[:, h]
            per_hand[hand_id] = {
                "valid_frames": int(np.count_nonzero(column)),
                "valid_ratio": round(float(np.mean(column)), 6) if column.size else 0.0,
                "missing_ratio": round(1.0 - float(np.mean(column)), 6) if column.size else 1.0,
                "depth_min_m": round(float(np.min(depth_valid)), 6) if depth_valid.size else None,
                "depth_median_m": round(float(np.median(depth_valid)), 6) if depth_valid.size else None,
                "depth_max_m": round(float(np.max(depth_valid)), 6) if depth_valid.size else None,
            }
        first = per_hand[next(iter(per_hand))] if per_hand else {}
        result.metrics.update(
            {
                "num_frames": sequence.num_frames,
                "num_hands": sequence.num_hands,
                "hand_ids": list(hand_ids),
                "selection": ctx.config.hand.selection,
                "valid_frames": int(np.count_nonzero(valid)),
                "valid_ratio": round(float(np.mean(valid)), 6) if valid.size else 0.0,
                "missing_ratio": round(1.0 - float(np.mean(valid)), 6) if valid.size else 1.0,
                "provider": provenance.get("provider", provider.name),
                "provider_tag": provider_tag,
                "frame_convention": provenance.get("frame_convention", "n/a"),
                "confidence_source": provenance.get("confidence_source", "unknown"),
                "depth_min_m": first.get("depth_min_m"),
                "depth_median_m": first.get("depth_median_m"),
                "depth_max_m": first.get("depth_max_m"),
                "per_hand": per_hand,
                "jointly_valid_frames": (
                    int(np.count_nonzero(np.all(valid, axis=1))) if valid.ndim == 2 else 0
                ),
            }
        )
        if "projection_check" in provenance:
            result.metrics["projection_check"] = provenance["projection_check"]
        if provenance.get("nan_mano_indices"):
            result.warnings.append(
                f"MANO indices {provenance['nan_mano_indices']} are not derivable from the "
                "ARKit skeleton and are stored as NaN (ADR-0003)"
            )
        result.payload["sequence"] = sequence
        result.payload["provenance"] = provenance
        return result

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        res = ValidationResult()
        seq = result.payload.get("sequence")
        if not isinstance(seq, HandSequence):
            res.add("hand.no_sequence", "stage produced no HandSequence payload")
            return res
        res.extend(seq.validate())
        res.extend(seq.validate_hand_identity())
        if res.ok:
            kp = np.asarray(seq["keypoints_C_m"])
            valid = np.asarray(seq["valid"])
            for h in range(valid.shape[1]):
                z = kp[:, h, 0, 2]
                # +z forward, and hands sit in front of an ego camera
                if valid[:, h].any() and not np.all(z[valid[:, h]] > 0.0):
                    res.add(
                        "hand.depth_sign",
                        f"hand {h}: wrist z <= 0 for at least one valid frame; the camera "
                        "optical frame requires +z forward (implementation-plan.md Gate M1)",
                    )
                    break
        for path in result.outputs:
            if not Path(path).is_file():
                res.add("hand.missing_output", f"declared output does not exist: {path}")
        return res
