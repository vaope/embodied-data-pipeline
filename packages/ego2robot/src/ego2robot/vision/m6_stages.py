"""M6 runtime stages: ``scene_depth`` then ``composite``.

``scene_depth`` turns the decoded (or inpainted) frames into a typed
:class:`~ego2robot.vision.scene_depth.SceneDepth` field and records the anchor test
that says whether its metres can be trusted.  ``composite`` applies the paper's
depth-aware rule to produce the synthetic video, plus the masks and decision map that
make every occlusion pixel explainable.

Both stages are ordinary pipeline stages: deterministic for a fixed input and
configuration, resumable, and manifest-emitting.  Neither of them decides that the
result is *correct* — the anchors and the review images are the evidence, and the user
is the acceptance authority (AGENTS.md rules 8 and 9).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core import artifacts as A
from ..core.capabilities import HandCapabilityError, require_sequence_support
from ..core.schema import CanonicalActionSequence, EpisodeManifest, ValidationResult
from ..pipeline.registry import register_stage
from ..pipeline.stage import StageContext, StageResult
from .composite import (
    DECISION_ARM,
    DECISION_GRIPPER_EXEMPT,
    DECISION_GRIPPER_OCCLUDED,
    DECISION_GRIPPER_VISIBLE,
    composite_frame,
    decision_legend,
)
from .da3_bridge import Da3BridgeError, run_da3
from .m5_preview import representative_indices, write_png, write_video
from .scene_depth import (
    SceneDepth,
    align_relative_depth,
    anchors_from_hand,
    anchors_from_tcp,
    depth_preview,
    validate_metric_scale,
)

__all__ = ["CompositeStage", "SceneDepthStage"]

#: Where a compositable background can live, in preference order.
_BACKGROUND_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("inpainted", ("inpainting", "frames", "background")),
    ("original", ("segmentation", "input")),
)

#: Where the hand mask (the occlusion exemption region) can live, in preference order.
_HAND_MASK_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("refined", ("segmentation", "bundle", "masks", "refined")),
    ("raw", ("segmentation", "bundle", "masks", "raw")),
    ("temporal-plate", ("previews", "m5-alternatives", "temporal-plate", "masks", "protected")),
)


def _frame_files(directory: Path) -> dict[int, Path]:
    """Map ``frame_index -> path`` for ``frame-%06d.png`` files."""
    out: dict[int, Path] = {}
    for path in sorted(directory.glob("frame-*.png")):
        stem = path.stem
        try:
            index = int(stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        out[index] = path
    return out


def _resolve_source(
    root: Path,
    table: tuple[tuple[str, tuple[str, ...]], ...],
    wanted: set[int],
    *,
    aspect: str,
) -> tuple[str, dict[int, Path]]:
    """First source that covers every wanted frame, else the best partial one."""
    best: tuple[str, dict[int, Path]] = ("none", {})
    for label, parts in table:
        directory = root.joinpath(*parts)
        if not directory.is_dir():
            continue
        frames = _frame_files(directory)
        if not frames:
            continue
        if wanted.issubset(frames.keys()):
            return label, frames
        if len(best[1]) < len(frames):
            best = (label, frames)
    return best


def _read_mask(path: Path) -> NDArray[np.bool_]:
    import imageio.v2 as imageio

    array = np.asarray(imageio.imread(path))
    if array.ndim == 3:
        array = array[..., 0]
    return np.asarray(array > 127, dtype=bool)


def _read_rgb(path: Path) -> NDArray[np.uint8]:
    import imageio.v2 as imageio

    array = np.asarray(imageio.imread(path))
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return np.ascontiguousarray(array[..., :3].astype(np.uint8))


@register_stage
class SceneDepthStage:
    """``scene_depth``: metric scene depth for the compositor, with an anchor test."""

    name = "scene_depth"
    version = "1"

    # -- contract ----------------------------------------------------------

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        res = ValidationResult()
        cfg = ctx.config.scene_depth
        if not cfg.enabled:
            return res
        if not ctx.layout.episode_json.is_file():
            res.add("scene_depth.no_episode", f"{ctx.layout.episode_json} missing; run 'ingest'")
            return res
        if not ctx.layout.camera_json.is_file():
            res.add("scene_depth.no_camera", f"{ctx.layout.camera_json} missing")
        if cfg.import_manifest:
            manifest = ctx.layout.root / cfg.import_manifest
            if not manifest.is_file():
                res.add(
                    "scene_depth.import_missing",
                    f"scene_depth.import_manifest points at a missing file: {manifest}",
                )
            return res
        for path, stage in (
            (ctx.layout.trajectory_npz, "robot"),
            (ctx.layout.actions_npz, "retarget"),
        ):
            if not path.is_file():
                res.add(
                    "scene_depth.missing_input",
                    f"{path} missing; run the {stage!r} stage first",
                )
        if ctx.layout.actions_npz.is_file():
            # The metric anchors are taken from hand 0's TCP (scene_depth.py).
            try:
                action = CanonicalActionSequence.load_npz(ctx.layout.actions_npz)
                require_sequence_support("scene_depth", action)
            except HandCapabilityError as exc:
                res.add("scene_depth.hand_count_unsupported", str(exc))
            except Exception as exc:  # noqa: BLE001
                res.add("scene_depth.unreadable_actions", str(exc))
        labels = [label for label, _ in _BACKGROUND_SOURCES]
        if cfg.input in labels:
            parts = dict(_BACKGROUND_SOURCES)[cfg.input]
            if not (ctx.layout.root.joinpath(*parts)).is_dir():
                res.add(
                    "scene_depth.no_background",
                    f"scene_depth.input={cfg.input!r} but {ctx.layout.root.joinpath(*parts)} "
                    "does not exist",
                )
        return res

    def fingerprint(self, ctx: StageContext) -> str:
        cfg = ctx.config.scene_depth
        robot_manifest = A.read_stage_manifest(ctx.layout.manifest_path("robot"))
        payload: dict[str, Any] = {
            "scene_depth": cfg.model_dump(mode="json"),
            "stage_sha": A.file_sha256(Path(__file__)),
            "core_sha": A.file_sha256(Path(__file__).with_name("scene_depth.py")),
            "bridge_sha": A.file_sha256(Path(__file__).with_name("da3_bridge.py")),
            "worker_sha": A.file_sha256(
                Path(__file__).resolve().parents[3] / "tools" / "da3_model_worker.py"
            ),
            "camera": A.file_sha256(ctx.layout.camera_json)
            if ctx.layout.camera_json.is_file()
            else "",
            "actions": A.file_sha256(ctx.layout.actions_npz)
            if ctx.layout.actions_npz.is_file()
            else "",
            "robot_self": (robot_manifest.input_fingerprints or {}).get("self", "")
            if robot_manifest
            else "",
        }
        wanted = set(int(v) for v in _trajectory_frames(ctx.layout).tolist())
        label, frames = _resolve_source(ctx.layout.root, _BACKGROUND_SOURCES, wanted, aspect="depth input")
        payload["background_source"] = label
        payload["background_hashes"] = {
            str(k): A.file_sha256(frames[k]) for k in sorted(frames) if k in wanted
        }
        return A.hash_json(payload)

    def run(self, ctx: StageContext) -> StageResult:
        result = StageResult(stage=self.name)
        cfg = ctx.config.scene_depth
        if not cfg.enabled:
            result.warnings.append("scene_depth.enabled is false; stage produced nothing")
            result.payload["skipped"] = True
            return result
        stage_dir = ctx.layout.stage_dir(self.name)
        stage_dir.mkdir(parents=True, exist_ok=True)
        episode = EpisodeManifest.model_validate(A.read_json(ctx.layout.episode_json))
        camera = episode.camera
        K = np.asarray(camera.K_array, dtype=np.float64)
        wanted = _trajectory_frames(ctx.layout)

        # -- 1. the field itself -------------------------------------------
        if cfg.import_manifest:
            imported = ctx.layout.root / cfg.import_manifest
            seq = SceneDepth.load(imported)
            source_label = f"imported:{cfg.import_manifest}"
            origin = {
                "kind": "imported",
                "manifest": str(imported),
                "manifest_sha256": A.file_sha256(imported),
            }
            result.warnings.append(
                f"scene depth was imported from {cfg.import_manifest}; no model was run"
            )
        else:
            source_label, frames = _resolve_source(
                ctx.layout.root, _BACKGROUND_SOURCES, set(wanted.tolist()), aspect="depth input"
            )
            if source_label == "none" or not frames:
                raise RuntimeError(
                    "no background frames found for the depth estimate; "
                    "run 'segmentation' (or provide segmentation/input) first"
                )
            missing = sorted(set(wanted.tolist()) - set(frames))
            if missing:
                raise RuntimeError(
                    f"background source {source_label!r} is missing frames {missing[:8]}"
                )
            seq, origin = self._estimate(
                ctx, stage_dir, frames, wanted, K, camera, source_label, "main"
            )

        if cfg.mode == "aligned" and seq.mode == "relative":
            anchors = _anchors(ctx, seq, source_label, K)
            seq, alignment = align_relative_depth(seq, anchors)
            origin["alignment"] = alignment
            result.metrics["alignment"] = alignment

        # -- 2. the anchor test --------------------------------------------
        anchors = _anchors(ctx, seq, seq.source, K)
        # On an inpainted background the hand region holds synthesised pixels, so an
        # anchor there would grade the inpainter rather than the depth model.  Drop
        # those anchors exactly where the compositor exempts the gripper.
        excluded = str(seq.source).startswith("inpainted")
        exclude_masks = _hand_region_masks(ctx, seq) if excluded else None
        report = validate_metric_scale(
            seq, anchors, cfg.validation, exclude_masks=exclude_masks
        )
        report["excluded_hand_region"] = bool(excluded)
        result.metrics["validation"] = report

        # -- 2b. reference scale check on frames that still show the hand ----
        # The TCP anchor is a coarse screen: the TCP sits between the fingers, so its
        # ray frequently misses the object it supposedly touches.  A hand keypoint is a
        # point on the hand surface with a known camera-frame z, which makes it real
        # metric evidence — but only while the hand is still in the image, so it needs
        # a small second pass on the decoded frames.
        reference_frames = int(getattr(cfg.validation, "reference_frames", 0) or 0)
        if reference_frames > 0 and not str(seq.source).startswith("original"):
            reference = self._reference_check(ctx, stage_dir, wanted, K, camera, reference_frames)
            if reference is not None:
                result.metrics["reference_check"] = reference
                status = reference.get("status")
                if status == "failed":
                    result.warnings.append(
                        "reference scale check against hand keypoints failed: median "
                        f"{reference.get('pooled', {}).get('median_abs_error_m')} m; the "
                        "metric depth scale is not verified on this episode"
                    )
                elif status == "passed":
                    result.metrics["scale_verified"] = True
        if report.get("status") == "failed":
            pooled = report.get("pooled", {})
            result.warnings.append(
                "scene depth anchor test failed: median "
                f"{pooled.get('median_abs_error_m')} m, p90 {pooled.get('p90_abs_error_m')} m "
                f"against thresholds {report.get('thresholds')}; the compositor's occlusion "
                "decisions are not trustworthy at this scale"
            )
        elif report.get("status") == "no_anchors":
            result.warnings.append(
                "no metric anchors were available, so the depth scale is unverified"
            )

        # -- 3. persist -----------------------------------------------------
        manifest = seq.save(stage_dir)
        result.add_output(manifest)

        previews = ctx.layout.previews_dir
        vmin, vmax = _depth_limits(seq)
        picks = representative_indices(seq.num_frames, maximum=3)
        made_previews: list[str] = []
        for pos in picks:
            frame_index = int(seq.frame_index[pos])
            rgb_path = _background_frame_path(ctx, seq.source, frame_index)
            if rgb_path is None:
                continue
            rgb = _read_rgb(rgb_path)
            path = depth_preview(
                rgb,
                seq.depth_m[pos],
                previews / f"scene-depth-frame-{frame_index:06d}.png",
                vmin=vmin,
                vmax=vmax,
                title=f"{episode.episode_id} frame {frame_index} · {seq.mode}",
            )
            result.add_output(path)
            made_previews.append(path.name)

        result.metrics.update(
            {
                "num_frames": seq.num_frames,
                "width": seq.width,
                "height": seq.height,
                "mode": seq.mode,
                "source": seq.source,
                "composable": seq.is_composable,
                "statistics": seq.statistics(),
                "previews": made_previews,
            }
        )
        result.payload["origin"] = origin
        result.payload["validation"] = report
        result.payload["scene_depth_manifest"] = str(manifest)
        return result

    def _estimate(
        self,
        ctx: StageContext,
        stage_dir: Path,
        frames: dict[int, Path],
        wanted: NDArray[np.int64],
        K: NDArray[np.float64],
        camera: Any,
        source_label: str,
        tag: str,
    ) -> tuple[SceneDepth, dict[str, Any]]:
        """Run the DA3 worker once and read its output back as a SceneDepth."""
        cfg = ctx.config.scene_depth
        model_mode = cfg.mode if cfg.mode != "aligned" else "relative"
        request_path = stage_dir / tag / "da3-request.json"
        result_path = stage_dir / tag / "da3-result.json"
        bridge_result = run_da3(
            runtime=cfg.runtime,
            model_dir=cfg.runtime.checkpoint_path or "",
            model_repo=cfg.runtime.repo_path or "",
            frames=[{"path": str(frames[int(fi)]), "frame_index": int(fi)} for fi in wanted],
            camera_intrinsics=np.asarray(K, dtype=float).tolist(),
            width=int(camera.width),
            height=int(camera.height),
            work_width=cfg.work_width,
            work_height=cfg.work_height,
            mode=model_mode,
            focal_scale_divisor=cfg.focal_scale_divisor,
            seed=cfg.seed,
            out_dir=stage_dir / tag / "model-output",
            request_path=request_path,
            result_path=result_path,
        )
        rows = bridge_result["frames"]
        depths = [np.load(Path(row["path"]), allow_pickle=False) for row in rows]
        confidences = [
            np.load(Path(row["confidence_path"]), allow_pickle=False)
            if row.get("confidence_path")
            else None
            for row in rows
        ]
        confidence = np.stack(confidences) if all(c is not None for c in confidences) else None
        seq = SceneDepth(
            depth_m=np.stack(depths),
            frame_index=np.asarray([int(r["frame_index"]) for r in rows], dtype=np.int64),
            mode=model_mode,
            source=source_label,
            confidence=confidence,
            provenance=bridge_result.get("provenance", {}),
        )
        origin = {
            "kind": "da3",
            "request": str(request_path),
            "result": str(result_path),
            "worker_log": str(result_path.with_suffix(".log")),
        }
        return seq, origin

    def _reference_check(
        self,
        ctx: StageContext,
        stage_dir: Path,
        wanted: NDArray[np.int64],
        K: NDArray[np.float64],
        camera: Any,
        count: int,
    ) -> dict[str, Any] | None:
        """Estimate depth on a few original frames and anchor it to the hand."""
        original = ctx.layout.root.joinpath(*dict(_BACKGROUND_SOURCES)["original"])
        if not original.is_dir():
            return None
        if not ctx.layout.hand_npz.is_file():
            return None
        files = _frame_files(original)
        usable = [int(fi) for fi in wanted if int(fi) in files]
        if len(usable) < 3:
            return None
        picks = representative_indices(len(usable), maximum=max(3, count))
        chosen = np.asarray([usable[p] for p in picks], dtype=np.int64)
        try:
            ref_seq, origin = self._estimate(
                ctx, stage_dir, files, chosen, K, camera, "original", "reference"
            )
        except Da3BridgeError as exc:
            return {"status": "unavailable", "reason": str(exc)[:400]}
        hand = np.load(ctx.layout.hand_npz)
        anchors = anchors_from_hand(hand, K, ref_seq.frame_index)
        report = validate_metric_scale(ref_seq, anchors, ctx.config.scene_depth.validation)
        report["frames"] = [int(v) for v in chosen]
        report["origin"] = origin
        report["note"] = (
            "independent scale evidence: depth re-estimated on frames that still contain "
            "the hand, so hand keypoints act as metric anchors"
        )
        ref_seq.save(stage_dir / "reference")
        return report

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        res = ValidationResult()
        if result.payload.get("skipped"):
            return res
        metrics = result.metrics
        if metrics.get("num_frames", 0) == 0:
            res.add("scene_depth.no_frames", "the scene depth field has no frames")
        if metrics.get("composable") is False:
            res.add(
                "scene_depth.relative_depth",
                "the field is mode='relative'; the compositor will refuse it (ADR-0008). "
                "Promote it with align_relative_depth or fix the mode.",
                severity="warning",
            )
        validation = metrics.get("validation", {})
        if validation.get("status") == "failed":
            res.add(
                "scene_depth.anchor_test_failed",
                "metric anchors disagree with the depth scale; see metrics.validation",
                severity="warning",
            )
        for path in result.outputs:
            if not Path(path).is_file():
                res.add("scene_depth.missing_output", f"declared output does not exist: {path}")
        return res


@register_stage
class CompositeStage:
    """``composite``: depth-aware robot overlay over the human-free background."""

    name = "composite"
    version = "1"

    # -- contract ----------------------------------------------------------

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        res = ValidationResult()
        cfg = ctx.config.composite
        if not cfg.enabled:
            return res
        for path, stage in (
            (ctx.layout.trajectory_npz, "robot"),
            (ctx.layout.actions_npz, "retarget"),
            (ctx.layout.depth_dir / "robot_depth_m.npz", "render"),
            (ctx.layout.visual_dir / "robot", "render"),
            (ctx.layout.manifest_path("render"), "render"),
        ):
            if not path.exists():
                res.add(
                    "composite.missing_input",
                    f"{path} missing; run the {stage!r} stage first",
                )
        if not ctx.layout.manifest_path("scene_depth").is_file():
            res.add(
                "composite.no_scene_depth",
                f"{ctx.layout.manifest_path('scene_depth')} missing; run the 'scene_depth' stage",
            )
        if ctx.layout.actions_npz.is_file():
            # compose_frame() takes one robot mask + one gripper mask, so two
            # hands/arms have no ownership decision yet (ADR-0022).
            try:
                action = CanonicalActionSequence.load_npz(ctx.layout.actions_npz)
                require_sequence_support("composite", action)
            except HandCapabilityError as exc:
                res.add("composite.hand_count_unsupported", str(exc))
            except Exception as exc:  # noqa: BLE001
                res.add("composite.unreadable_actions", str(exc))
        return res

    def fingerprint(self, ctx: StageContext) -> str:
        cfg = ctx.config.composite
        render_manifest = A.read_stage_manifest(ctx.layout.manifest_path("render"))
        depth_manifest = A.read_stage_manifest(ctx.layout.manifest_path("scene_depth"))
        scene_dir = ctx.layout.stage_dir("scene_depth")
        payload: dict[str, Any] = {
            "composite": cfg.model_dump(mode="json"),
            "stage_sha": A.file_sha256(Path(__file__)),
            "core_sha": A.file_sha256(Path(__file__).with_name("composite.py")),
            "depth_core_sha": A.file_sha256(Path(__file__).with_name("scene_depth.py")),
            "render_self": (render_manifest.input_fingerprints or {}).get("self", "")
            if render_manifest
            else "",
            "scene_self": (depth_manifest.input_fingerprints or {}).get("self", "")
            if depth_manifest
            else "",
            "scene_depth_manifest": A.file_sha256(scene_dir / "scene-depth.json")
            if (scene_dir / "scene-depth.json").is_file()
            else "",
        }
        wanted = set(_trajectory_frames(ctx.layout).tolist())
        bg_label, bg = _resolve_source(ctx.layout.root, _BACKGROUND_SOURCES, wanted, aspect="background")
        hand_label, hand = _resolve_source(ctx.layout.root, _HAND_MASK_SOURCES, wanted, aspect="hand mask")
        payload["background_source"] = bg_label
        payload["hand_mask_source"] = hand_label
        payload["inputs"] = {
            "background": {str(k): A.file_sha256(bg[k]) for k in sorted(bg) if k in wanted},
            "hand_mask": {str(k): A.file_sha256(hand[k]) for k in sorted(hand) if k in wanted},
        }
        return A.hash_json(payload)

    def run(self, ctx: StageContext) -> StageResult:
        result = StageResult(stage=self.name)
        cfg = ctx.config.composite
        if not cfg.enabled:
            result.warnings.append("composite.enabled is false; stage produced nothing")
            result.payload["skipped"] = True
            return result

        stage_dir = ctx.layout.stage_dir(self.name)
        stage_dir.mkdir(parents=True, exist_ok=True)
        episode = EpisodeManifest.model_validate(A.read_json(ctx.layout.episode_json))
        trajectory = np.load(ctx.layout.trajectory_npz)
        frame_index = np.asarray(trajectory["frame_index"], dtype=np.int64)
        wanted = frame_index.tolist()
        wanted_set = set(wanted)

        scene_dir = ctx.layout.stage_dir("scene_depth")
        seq = SceneDepth.load(scene_dir / "scene-depth.json")
        if cfg.mode == "depth_aware":
            seq.assert_composable("composite")

        robot_depth = np.load(ctx.layout.depth_dir / "robot_depth_m.npz")["depth_m"]
        robot_index = np.asarray(
            np.load(ctx.layout.depth_dir / "robot_depth_m.npz")["frame_index"], dtype=np.int64
        )

        bg_label, bg = _resolve_source(ctx.layout.root, _BACKGROUND_SOURCES, wanted_set, aspect="background")
        if bg_label == "none":
            raise RuntimeError("no background frames available to composite onto")
        hand_label, hand = _resolve_source(ctx.layout.root, _HAND_MASK_SOURCES, wanted_set, aspect="hand mask")
        if hand_label == "none":
            result.warnings.append(
                "no hand mask found; the gripper has no occlusion exemption and will be "
                "hidden wherever the scene depth says it is behind, including at the grasp point"
            )
        elif not wanted_set.issubset(set(hand.keys())):
            result.warnings.append(
                f"hand mask source {hand_label!r} covers only "
                f"{len(wanted_set & set(hand.keys()))}/{len(wanted)} frames; "
                "missing frames run without the exemption"
            )
        if bg_label == "original":
            result.warnings.append(
                "compositing over the decoded source frames, not the inpainted background: "
                "the human arm is still visible in the output"
            )

        prefix = "frames" if cfg.mode == "depth_aware" else "frames-alpha"
        out_frames = stage_dir / prefix
        out_decision = stage_dir / "decision" if cfg.mode == "depth_aware" else stage_dir / "decision-alpha"
        out_visible = stage_dir / "masks" / "visible"
        out_occluded = stage_dir / "masks" / "occluded"

        totals: dict[str, int] = {}
        per_frame: list[dict[str, Any]] = []
        video_frames: list[NDArray[np.uint8]] = []
        fps = _fps(episode)
        limit = cfg.hand_mask_dilation_kernel
        for pos, fi in enumerate(wanted):
            bg_path = bg.get(int(fi))
            if bg_path is None:
                raise RuntimeError(f"background source {bg_label!r} is missing frame {fi}")
            background = _read_rgb(bg_path)
            robot_rgb_path = ctx.layout.visual_dir / "robot" / f"frame-{int(fi):06d}.png"
            robot_mask_path = ctx.layout.masks_dir / "robot" / f"frame-{int(fi):06d}.png"
            gripper_mask_path = ctx.layout.masks_dir / "gripper" / f"frame-{int(fi):06d}.png"
            if not (robot_rgb_path.is_file() and robot_mask_path.is_file() and gripper_mask_path.is_file()):
                raise RuntimeError(f"render artifacts are missing for frame {fi}")
            depth_pos = int(np.searchsorted(robot_index, int(fi)))
            if depth_pos >= robot_index.size or int(robot_index[depth_pos]) != int(fi):
                raise RuntimeError(f"robot depth has no entry for frame {fi}")
            scene = seq.frame(int(fi))
            if scene is None:
                raise RuntimeError(f"scene depth has no entry for frame {fi}")
            hand_mask = _read_mask(hand[int(fi)]) if int(fi) in hand else None
            composed = composite_frame(
                background,
                _read_rgb(robot_rgb_path),
                _read_mask(robot_mask_path),
                _read_mask(gripper_mask_path),
                robot_depth[depth_pos],
                scene,
                scene_mode=seq.mode,
                hand_mask=hand_mask,
                mode="depth_aware" if cfg.mode == "depth_aware" else "debug_alpha",
                hand_mask_dilation_kernel=int(limit),
                arm_body_always_visible=bool(cfg.arm_body_always_visible),
            )
            write_png(out_frames / f"frame-{int(fi):06d}.png", composed.image)
            write_png(out_decision / f"frame-{int(fi):06d}.png", composed.decision_rgb())
            if cfg.mode == "depth_aware":
                write_png(out_visible / f"frame-{int(fi):06d}.png", composed.visible_mask)
                write_png(out_occluded / f"frame-{int(fi):06d}.png", composed.occluded_mask)
            for key, value in composed.metrics.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
            per_frame.append({"frame_index": int(fi), **composed.metrics})
            video_frames.append(composed.image)

        if video_frames:
            video_name = "composite.mp4" if cfg.mode == "depth_aware" else "composite-alpha.mp4"
            video_path = write_video(stage_dir / video_name, video_frames, fps=fps)
            result.add_output(video_path)
        review = self._review_sheet(ctx, prefix, out_decision, wanted, result)

        n_gripper = max(totals.get("gripper_pixels", 0), 1)
        result.metrics.update(
            {
                "num_frames": len(per_frame),
                "mode": cfg.mode,
                "background_source": bg_label,
                "hand_mask_source": hand_label,
                "scene_depth_mode": seq.mode,
                "hand_mask_dilation_kernel": int(limit),
                "arm_pixels_total": totals.get("arm_pixels", 0),
                "gripper_pixels_total": totals.get("gripper_pixels", 0),
                "gripper_occluded_fraction": round(
                    totals.get("gripper_occluded_pixels", 0) / n_gripper, 6
                ),
                "gripper_exempt_fraction": round(
                    totals.get("gripper_exempt_pixels", 0) / n_gripper, 6
                ),
                "gripper_scene_depth_unknown_fraction": round(
                    totals.get("gripper_scene_depth_unknown_pixels", 0) / n_gripper, 6
                ),
                "review_sheet": str(review) if review else None,
                "decision_legend": decision_legend(),
                "valid_episode": cfg.mode == "depth_aware" and seq.is_composable,
            }
        )
        result.payload["frames"] = per_frame
        if cfg.mode != "depth_aware":
            result.warnings.append(
                "debug alpha composite: the robot is never occluded by the scene, so this "
                "output cannot be part of a valid episode (ADR-0008)"
            )
        return result

    def _review_sheet(
        self,
        ctx: StageContext,
        prefix: str,
        decision_dir: Path,
        wanted: list[int],
        result: StageResult,
    ) -> Path | None:
        """Background | robot render | composite | decision, one row per sampled frame."""
        from PIL import Image, ImageDraw

        from .scene_depth import _load_font

        picks = representative_indices(len(wanted), maximum=4)
        stage_dir = ctx.layout.stage_dir(self.name)
        rows: list[NDArray[np.uint8]] = []
        for pos in picks:
            fi = int(wanted[pos])
            composite_path = stage_dir / prefix / f"frame-{fi:06d}.png"
            decision_path = decision_dir / f"frame-{fi:06d}.png"
            _label, bg = _resolve_source(
                ctx.layout.root, _BACKGROUND_SOURCES, {fi}, aspect="background"
            )
            if not composite_path.is_file() or fi not in bg:
                continue
            panels = [
                _read_rgb(bg[fi]),
                _read_rgb(ctx.layout.visual_dir / "robot" / f"frame-{fi:06d}.png"),
                _read_rgb(composite_path),
                _read_rgb(decision_path) if decision_path.is_file() else np.zeros((8, 8, 3), np.uint8),
            ]
            height = panels[0].shape[0]
            width = panels[0].shape[1]
            scale = min(1.0, 520.0 / float(width))
            if scale < 1.0:
                resized = [
                    np.asarray(
                        Image.fromarray(p).resize(
                            (int(p.shape[1] * scale), int(p.shape[0] * scale)),
                            Image.Resampling.LANCZOS,
                        )
                    )
                    for p in panels
                ]
            else:
                resized = panels
            rows.append(np.concatenate(resized, axis=1))
        if not rows:
            return None
        sheet = np.concatenate(rows, axis=0)
        image = Image.fromarray(sheet)
        draw = ImageDraw.Draw(image)
        font = _load_font(max(16, int(sheet.shape[1] / 70)))
        labels = ["background", "robot render", "composite", "decision map"]
        panel_w = sheet.shape[1] // 4
        for i, label in enumerate(labels):
            draw.rectangle([i * panel_w, 0, i * panel_w + 12 * len(label), 26], fill=(12, 14, 18))
            draw.text((i * panel_w + 6, 3), label, font=font, fill=(228, 232, 240))
        path = ctx.layout.previews_dir / f"composite-review-{prefix}.png"
        write_png(path, np.asarray(image))
        result.add_output(path)
        return path

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        res = ValidationResult()
        if result.payload.get("skipped"):
            return res
        metrics = result.metrics
        if metrics.get("num_frames", 0) == 0:
            res.add("composite.no_frames", "no frames were composited")
        if metrics.get("arm_pixels_total", 0) == 0:
            res.add(
                "composite.no_arm",
                "the robot arm never had a pixel; the render or base placement is wrong",
            )
        if metrics.get("scene_depth_mode") not in {"metric", "aligned"}:
            res.add(
                "composite.non_metric_depth",
                f"scene depth mode is {metrics.get('scene_depth_mode')!r}; "
                "this output cannot be a valid episode (ADR-0008)",
                severity="warning",
            )
        for path in result.outputs:
            if not Path(path).is_file():
                res.add("composite.missing_output", f"declared output does not exist: {path}")
        return res


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _trajectory_frames(layout: Any) -> NDArray[np.int64]:
    payload = np.load(layout.trajectory_npz)
    return np.asarray(payload["frame_index"], dtype=np.int64)


def _anchors(ctx: StageContext, seq: SceneDepth, source_label: str, K: NDArray[np.float64]):
    """Metric anchors that are meaningful for the depth input that was used."""
    actions = CanonicalActionSequence.load_npz(ctx.layout.actions_npz)
    anchors = [anchors_from_tcp(actions, K, seq.frame_index)]
    # Hand keypoints are only anchors while the hand is still in the depth input.
    if not str(source_label).startswith(("inpainted", "imported")):
        hand = None
        if ctx.layout.hand_npz.is_file():
            try:
                hand = np.load(ctx.layout.hand_npz)
            except Exception:  # noqa: BLE001 - anchors are diagnostics, not inputs
                hand = None
        if hand is not None:
            anchors.append(anchors_from_hand(hand, K, seq.frame_index))
    return anchors


def _hand_region_masks(ctx: StageContext, seq: SceneDepth) -> dict[int, NDArray[np.bool_]]:
    """Per-frame dilated hand masks, used to drop anchors over inpainted pixels."""
    wanted = set(int(v) for v in seq.frame_index.tolist())
    _label, files = _resolve_source(ctx.layout.root, _HAND_MASK_SOURCES, wanted, aspect="hand mask")
    kernel = int(ctx.config.composite.hand_mask_dilation_kernel)
    if not files:
        return {}
    from scipy.ndimage import binary_dilation

    structure = np.ones((kernel, kernel), dtype=bool) if kernel > 1 else None
    masks: dict[int, NDArray[np.bool_]] = {}
    for frame_index in wanted:
        path = files.get(frame_index)
        if path is None:
            continue
        mask = _read_mask(path)
        masks[frame_index] = (
            binary_dilation(mask, structure=structure) if structure is not None else mask
        )
    return masks


def _background_frame_path(ctx: StageContext, source_label: str, frame_index: int) -> Path | None:
    table = dict(_BACKGROUND_SOURCES)
    label = source_label if source_label in table else "original"
    parts = table.get(label)
    if not parts:
        return None
    path = ctx.layout.root.joinpath(*parts) / f"frame-{frame_index:06d}.png"
    return path if path.is_file() else None


def _depth_limits(seq: SceneDepth) -> tuple[float, float]:
    finite = seq.depth_m[np.isfinite(seq.depth_m)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if not (hi > lo):
        hi = lo + 0.1
    return lo, hi


def _fps(episode: EpisodeManifest) -> float:
    try:
        fps = float(getattr(episode, "fps", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        fps = 0.0
    return fps if fps > 0 else 30.0
