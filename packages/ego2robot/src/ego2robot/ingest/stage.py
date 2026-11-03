"""Ingest stage: probe the video, lock the timebase, publish ``episode.json``.

Corresponds to implementation-plan.md M1 ("视频 probe / episode.json /
camera.json") and system-design.md 5.3 (PTS is the master clock) and 9.1
(camera provenance enum).

Inputs understood by this stage (see ``docs/fixture-format.md``):

* ``<episode_dir>/source/video.mp4`` + ``source/hands.hdf5``  -- project fixture
* ``<episode_dir>/source/<name>.mp4`` + ``<name>.hdf5``       -- EgoDex episode
* ``episode.json`` already present                           -- validate only
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core import artifacts as A
from ..core.frames import Frame, TransformName, transform_name
from ..core.schema import (
    CameraFile,
    CameraModel,
    EpisodeManifest,
    EpisodeSource,
    Province,
    TaskSpec,
    Timebase,
    ValidationResult,
)
from ..pipeline.registry import register_stage
from ..pipeline.stage import StageContext, StageResult
from .egodex import EGODEX_DEFAULT_CAMERA, is_egodex_hdf5, probe_egodex_hdf5
from .fixture import list_hands
from .video import probe_video

__all__ = ["IngestStage", "discover_sources"]

_VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def discover_sources(episode_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Find ``(video, hdf5, camera_json)`` in an episode directory."""
    src = episode_dir / "source"
    search_dirs = [src, episode_dir]
    video: Path | None = None
    hdf5: Path | None = None
    camera_json: Path | None = None
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file():
                if video is None and f.suffix.lower() in _VIDEO_SUFFIXES:
                    video = f
                elif hdf5 is None and f.suffix.lower() == ".hdf5":
                    hdf5 = f
                elif camera_json is None and f.name == "camera.json":
                    camera_json = f
    return video, hdf5, camera_json


def _build_camera(
    *,
    width: int,
    height: int,
    camera_json: Path | None,
    hdf5: Path | None,
) -> tuple[CameraModel, dict[str, Any]]:
    """Resolve the camera model with an explicit provenance label.

    Precedence: the episode's own ``source/camera.json`` (which a dataset adapter
    or the fixture builder wrote), then the dataset's stored intrinsics, then the
    published EgoDex intrinsics labelled ``assumed`` so nothing can enter a
    formal dataset by accident (system-design.md 9.1).
    """
    import json

    notes: dict[str, Any] = {}
    if camera_json is not None:
        with open(camera_json, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if "camera" in payload:
            camera_file = CameraFile.model_validate(payload)
            cam = camera_file.camera
            notes["camera_source"] = f"source/camera.json ({cam.calibration_source})"
            notes["camera_file_provenance"] = camera_file.provenance
        else:
            # tolerate a bare CameraModel document for hand-written fixtures
            cam = CameraModel.model_validate(payload)
            notes["camera_source"] = f"source/camera.json bare model ({cam.calibration_source})"
        return cam, notes
    if hdf5 is not None and is_egodex_hdf5(hdf5):
        probe = probe_egodex_hdf5(hdf5)
        if probe.camera_intrinsic is not None:
            cam = CameraModel(
                width=width,
                height=height,
                K=np.asarray(probe.camera_intrinsic, dtype=np.float64).tolist(),
                distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
                calibration_source="dataset",
                confidence=0.9,
            )
            notes["camera_source"] = "EgoDex /camera/intrinsic"
            return cam, notes
    # Last resort: the published EgoDex intrinsics, labelled 'assumed' so the
    # model refuses to let them enter a formal dataset silently.
    cam = CameraModel(
        width=width or EGODEX_DEFAULT_CAMERA["width"],
        height=height or EGODEX_DEFAULT_CAMERA["height"],
        K=EGODEX_DEFAULT_CAMERA["K"],
        calibration_source="assumed",
        confidence=0.2,
        debug_only=True,
    )
    notes["camera_source"] = "EgoDex published intrinsics, assumed for this video resolution"
    notes["warning"] = "camera parameters are assumed; this episode is debug-only"
    return cam, notes


@register_stage
class IngestStage:
    """``ingest``: source discovery, video probe, timebase and camera lock."""

    name = "ingest"
    version = "1"

    # -- contract ----------------------------------------------------------

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        res = ValidationResult()
        video, hdf5, _camera = discover_sources(ctx.layout.root)
        if video is None:
            res.add(
                "ingest.no_video",
                f"no video file found under {ctx.layout.root / 'source'}; supported "
                f"suffixes: {', '.join(_VIDEO_SUFFIXES)}",
            )
        if hdf5 is None:
            res.add(
                "ingest.no_hdf5",
                f"no *.hdf5 hand annotation found under {ctx.layout.root / 'source'}; "
                "Path A requires paired annotations",
                severity="warning",
            )
        return res

    def fingerprint(self, ctx: StageContext) -> str:
        video, hdf5, camera_json = discover_sources(ctx.layout.root)
        inputs: dict[str, str] = {"config": A.hash_json(ctx.config.model_dump(mode="json"))}
        if video is not None:
            inputs["video"] = A.file_sha256(video)
        if hdf5 is not None:
            inputs["hdf5"] = A.file_sha256(hdf5)
        if camera_json is not None:
            inputs["camera_json"] = A.file_sha256(camera_json)
        inputs["video_size"] = str(video.stat().st_size if video else 0)
        return A.hash_json(inputs)

    def run(self, ctx: StageContext) -> StageResult:
        result = StageResult(stage=self.name)
        log = ctx.log
        layout = ctx.layout

        video, hdf5, camera_json = discover_sources(layout.root)
        if video is None:
            raise FileNotFoundError(f"no video under {layout.source_dir}")

        cfg = ctx.config.ingest
        probe = probe_video(video, decode_check_frames=cfg.decode_check_frames)
        if log:
            log.info(
                f"probed {video.name}: {probe.width}x{probe.height} {probe.num_frames} frames "
                f"codec={probe.stream.codec} backend={probe.backend} "
                f"median fps={probe.measured_fps:.3f}"
            )

        if cfg.require_monotonic_timestamps and not probe.monotonic:
            raise ValueError(
                f"{video}: container PTS is not strictly increasing; refusing to build a "
                "timebase that would silently reorder frames (system-design.md 5.3)"
            )
        if probe.backend != "pyav":
            result.warnings.append(
                "video was probed with the OpenCV fallback: timestamps are "
                "frame_index/nominal_fps, not container PTS"
            )
        result.warnings.extend(probe.warnings)

        num_frames = probe.num_frames
        if cfg.max_frames is not None:
            num_frames = min(num_frames, cfg.max_frames)

        timestamps = [float(t) for t in probe.timestamps_s[:num_frames]]
        if timestamps:
            t0 = timestamps[0]
            timestamps = [t - t0 for t in timestamps]

        fixture_camera = None
        camera_path = camera_json
        if not camera_path:
            candidate = layout.source_dir / "camera.json"
            if candidate.is_file():
                camera_path = candidate

        camera, camera_notes = _build_camera(
            width=probe.width,
            height=probe.height,
            camera_json=camera_path,
            hdf5=hdf5,
        )
        if camera_notes.get("warning"):
            result.warnings.append(str(camera_notes["warning"]))

        # ---- task instruction & hand annotations --------------------------
        instruction = ""
        source_dataset = "local"
        hand_summary: dict[str, Any] = {}
        if hdf5 is not None:
            if is_egodex_hdf5(hdf5):
                egoprobe = probe_egodex_hdf5(hdf5)
                instruction = egoprobe.instruction
                source_dataset = "egodex"
                hand_summary = egoprobe.summary()
            else:
                source_dataset = "fixture"
                groups = list_hands(hdf5)
                hand_summary = {
                    "hdf5": str(hdf5),
                    "format": "ego2robot fixture",
                    "hand_groups": groups,
                }
                case_json = layout.root / "case.json"
                if case_json.is_file():
                    case = A.read_json(case_json)
                    instruction = str(case.get("instruction", ""))
                    hand_summary["case_id"] = case.get("case_id")

        name = transform_name(Frame.C, Frame.B)

        # ``T_C_B`` is an operator/dataset decision, not something ingest may
        # invent or drop.  When the episode already declares one (a fixture, or a
        # hand-written manifest), carry it forward unchanged.
        preserved_T_C_B: list[list[float]] | None = None
        preserved_notes: list[str] = []
        if layout.episode_json.is_file():
            try:
                previous = EpisodeManifest.model_validate(A.read_json(layout.episode_json))
            except Exception:  # noqa: BLE001 - a broken previous manifest is replaced
                previous = None
                result.warnings.append(
                    "the existing episode.json could not be parsed; it was rebuilt from source"
                )
            if previous is not None and previous.T_C_B is not None:
                preserved_T_C_B = [list(row) for row in previous.T_C_B]
                result.metrics["preserved_T_C_B"] = preserved_T_C_B
                preserved_notes.append("T_C_B preserved from the existing episode.json")

        manifest = EpisodeManifest(
            episode_id=ctx.episode_id,
            source=EpisodeSource(
                video_uri=str(video.relative_to(layout.root)).replace("\\", "/"),
                sha256=A.file_sha256(video),
                source_dataset=source_dataset,
                hdf5_uri=str(hdf5.relative_to(layout.root)).replace("\\", "/") if hdf5 else None,
                hdf5_sha256=A.file_sha256(hdf5) if hdf5 else None,
                license="CC-BY-NC-ND" if source_dataset == "egodex" else "project-derived",
                publishable=False if source_dataset == "egodex" else True,
            ),
            timebase=Timebase(
                num_frames=num_frames,
                nominal_fps=probe.nominal_fps or probe.measured_fps,
                timestamps_s=timestamps,
                timebase_source="pts" if probe.backend == "pyav" else "nominal_fps",
            ),
            camera=camera,
            task=TaskSpec(instruction=instruction),
            provenance=Province(
                config_hash=ctx.config_hash,
                code_revision="",
                created_at=A.utc_now_iso(),
            ),
            T_C_B=preserved_T_C_B,
            notes=[
                f"video probe backend: {probe.backend}",
                f"camera provenance: {camera_notes.get('camera_source', 'unknown')}",
                f"camera frame convention: {Frame.C.value} (+x right, +y down, +z forward)",
                f"robot base frame {name.target.value} is declared but not yet placed"
                if preserved_T_C_B is None
                else f"robot base placement T_C_B was already declared by the source",
                *preserved_notes,
            ],
        )
        episode_path = A.atomic_write_json(layout.episode_json, manifest)
        result.add_output(episode_path)

        camera_payload = CameraFile(
            camera=camera,
            camera_frame=Frame.C.value,
            provenance=str(camera_notes.get("camera_source", "unknown")),
        )
        result.add_output(A.atomic_write_json(layout.camera_json, camera_payload))

        if probe.first_frame_sha256:
            result.metrics["first_frame_sha256"] = probe.first_frame_sha256
        result.metrics.update(
            {
                "num_frames": num_frames,
                "width": probe.width,
                "height": probe.height,
                "codec": probe.stream.codec,
                "probe_backend": probe.backend,
                "nominal_fps": round(probe.nominal_fps, 6),
                "measured_fps": round(probe.measured_fps, 6),
                "duration_s": round(timestamps[-1], 6) if len(timestamps) > 1 else 0.0,
                "pts_monotonic": probe.monotonic,
                "camera_calibration_source": camera.calibration_source,
                "fx": round(camera.fx, 4),
                "fy": round(camera.fy, 4),
                "cx": round(camera.cx, 4),
                "cy": round(camera.cy, 4),
                "source_dataset": source_dataset,
            }
        )
        if hand_summary:
            result.metrics["hand_annotations"] = hand_summary

        result.payload["probe"] = probe
        result.payload["episode"] = manifest
        result.payload["camera"] = camera
        if hdf5 is not None:
            if hdf5.suffix.lower() != ".hdf5":
                available = []
            elif is_egodex_hdf5(hdf5):
                probe = probe_egodex_hdf5(hdf5)
                available = [
                    hand
                    for hand, joint in (("left", "leftHand"), ("right", "rightHand"))
                    if joint in probe.joint_names
                ]
            else:
                available = list_hands(hdf5)
            result.payload["hand_groups"] = available
        return result

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        res = ValidationResult()
        if not result.outputs:
            res.add("ingest.no_outputs", "ingest produced no artifacts")
        for path in result.outputs:
            if not Path(path).is_file():
                res.add("ingest.missing_output", f"declared output does not exist: {path}")
        episode = result.payload.get("episode")
        if isinstance(episode, EpisodeManifest):
            if episode.timebase.num_frames <= 0:
                res.add("ingest.empty_timebase", "timebase.num_frames must be positive")
            elif episode.timebase.timestamps_s is not None:
                arr = np.asarray(episode.timebase.timestamps_s)
                if arr.size != episode.timebase.num_frames:
                    res.add(
                        "ingest.timebase_length",
                        f"timestamps length {arr.size} != num_frames {episode.timebase.num_frames}",
                    )
        return res


def _unused() -> TransformName:  # pragma: no cover - documentation anchor
    """Keeps the ``T_C_B`` naming rule visible in this module."""
    return transform_name(Frame.C, Frame.B)
