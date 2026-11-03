"""M5 human-arm segmentation and background inpainting stages.

The model boundary is the isolated :mod:`m5_bridge` process. This module owns
episode inputs, PTS-aligned artifacts, validation, previews, and stage
fingerprints; it does not import model frameworks.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core import artifacts as A
from ..core.capabilities import require_sequence_support
from ..core.schema import EpisodeManifest, HandSequence, ValidationResult
from ..pipeline.registry import register_stage
from ..pipeline.stage import StageContext, StageResult
from .m5_contracts import (
    BackgroundFrame,
    BackgroundSequenceIndex,
    MaskFrame,
    MaskSequenceIndex,
    ModelRequest,
    ModelResult,
    RuntimeConfig,
    SegmentationConfig,
    SourceFrame,
    SourceFrameIndex,
)
from .mask_refine import refine_masks
from .m5_preview import (
    make_area_curve,
    make_inpainting_triptych,
    make_segmentation_triptych,
    representative_indices,
    status_image,
    write_png,
    write_video,
)

__all__ = ["InpaintingStage", "M5StageError", "SegmentationStage"]


class M5StageError(RuntimeError):
    """A stage failure that still exposes labelled, user-viewable diagnostics."""

    def __init__(self, message: str, *, diagnostic_outputs: tuple[Path, ...] = ()) -> None:
        super().__init__(message)
        self.diagnostic_outputs = diagnostic_outputs


def _episode_relative(layout: A.EpisodeLayout, relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(relative)
    if (
        not relative.strip()
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
    ):
        raise ValueError(f"episode artifact path must be relative and contained: {relative!r}")
    path = layout.root.joinpath(*posix.parts).resolve()
    try:
        path.relative_to(layout.root.resolve())
    except ValueError as exc:
        raise ValueError(f"episode artifact path escapes the episode: {relative!r}") from exc
    return path


def _episode_manifest(layout: A.EpisodeLayout, episode_id: str) -> EpisodeManifest:
    if not layout.episode_json.is_file():
        raise FileNotFoundError(f"episode manifest missing: {layout.episode_json}")
    episode = EpisodeManifest.model_validate(A.read_json(layout.episode_json))
    if episode.episode_id != episode_id:
        raise ValueError(f"episode.json id {episode.episode_id!r} != run id {episode_id!r}")
    return episode


def _require_succeeded_manifest(layout: A.EpisodeLayout, stage_name: str):
    path = layout.manifest_path(stage_name)
    manifest = A.read_stage_manifest(path)
    if manifest is None or manifest.stage != stage_name or manifest.status != "succeeded":
        raise ValueError(f"{stage_name} stage manifest is missing or not succeeded: {path}")
    missing = [rel for rel in manifest.outputs if not (layout.root / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"{stage_name} manifest outputs are missing: {missing[:5]}")
    return manifest


def _approved_m3_record(
    ctx: StageContext, episode: EpisodeManifest, cfg: SegmentationConfig
) -> tuple[Path, dict[str, Any]]:
    record_path = _episode_relative(ctx.layout, cfg.approval_record)
    if not record_path.is_file():
        raise FileNotFoundError(f"M3 review record is missing: {record_path}")
    # PowerShell-authored UTF-8 JSON can carry a BOM.
    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    if record.get("accepted") is not True:
        raise ValueError(f"M3 review is not accepted: {record_path}")
    if record.get("scope") != "episode_projection_video":
        raise ValueError("M3 review scope must be 'episode_projection_video'")
    if record.get("episode_id") != episode.episode_id:
        raise ValueError("M3 review episode_id does not match the current episode")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("M3 review record has no evidence list")
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"M3 review evidence item {index} has no relative path")
        path = _episode_relative(ctx.layout, item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"M3 review evidence is missing: {path}")
        actual = A.file_sha256(path)
        expected = item.get("sha256")
        if actual != expected:
            raise ValueError(f"M3 review evidence checksum mismatch: {item['path']}")
    return record_path, record


def _source_video_path(ctx: StageContext, episode: EpisodeManifest) -> Path:
    uri = str(episode.source.video_uri)
    path = Path(uri)
    if not path.is_absolute():
        path = ctx.layout.root / path
    path = path.resolve()
    try:
        path.relative_to(ctx.layout.root.resolve())
    except ValueError as exc:
        raise ValueError("episode video_uri must resolve to a file inside the episode directory") from exc
    if not path.is_file():
        raise FileNotFoundError(f"episode source video is missing: {path}")
    return path


def _approved_rig_record(
    ctx: StageContext, episode: EpisodeManifest, cfg: SegmentationConfig
) -> tuple[Path, dict[str, Any]]:
    """The two-arm equivalent of the M3 video review (ADR-0026).

    A bimanual episode never has a succeeded ``projection`` stage (single-arm) and
    therefore never has ``quality/reviews/m3-video.json``.  What it does have is a
    two-arm rig: per-arm IK/FK plus rendered frames from the real camera pose.  The
    user reviews *that* instead, and the record binds the geometry record and the
    sampled renders by SHA-256 so a re-run that changes either invalidates the
    approval instead of silently reusing it.

    ``reviewer`` must be literally ``user``: an agent may not write this file on the
    user's behalf (AGENTS.md rules 2 and 9).
    """
    record_path = _episode_relative(ctx.layout, cfg.rig_approval_record)
    if not record_path.is_file():
        raise FileNotFoundError(
            f"two-arm rig review record is missing: {record_path}; the user must review "
            "robot/rig/rig.json plus its rendered frames and approve them explicitly "
            "(ADR-0026)"
        )
    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    if record.get("schema_version") != "ego2robot.rig.review.v1":
        raise ValueError("rig review record must declare schema_version 'ego2robot.rig.review.v1'")
    if record.get("accepted") is not True:
        raise ValueError(f"rig review is not accepted: {record_path}")
    if record.get("scope") != "episode_rig_geometry":
        raise ValueError("rig review scope must be 'episode_rig_geometry'")
    if record.get("episode_id") != episode.episode_id:
        raise ValueError("rig review episode_id does not match the current episode")
    if record.get("reviewer") != "user":
        raise ValueError(
            "rig review reviewer must be 'user'; an agent may not approve geometry evidence "
            "on the user's behalf (AGENTS.md rules 2 and 9)"
        )
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("rig review record has no evidence list")
    seen: list[str] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"rig review evidence item {index} has no relative path")
        path = _episode_relative(ctx.layout, item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"rig review evidence is missing: {path}")
        if A.file_sha256(path) != item.get("sha256"):
            raise ValueError(f"rig review evidence checksum mismatch: {item['path']}")
        seen.append(path.relative_to(ctx.layout.root).as_posix())
    rig_record = ctx.layout.rig_json.relative_to(ctx.layout.root).as_posix()
    if rig_record not in seen:
        raise ValueError(f"rig review must bind {rig_record}; otherwise it approves nothing")
    if not any(name.startswith("robot/rig/render/") and name.endswith(".png") for name in seen):
        raise ValueError(
            "rig review must bind at least one rendered frame under robot/rig/render/; "
            "geometry numbers alone are not a visual review"
        )
    return record_path, record


#: Stages that can serve as the reviewed visual upstream of the human-removal steps.
#:
#: ``projection`` is the single-arm chain (M3).  ``rig`` is the two-arm chain
#: (ADR-0022/0025) and is the only upstream a ``H=2`` episode can have.
VISUAL_UPSTREAM_STAGES = ("rig", "projection")


def _visual_upstream(layout: A.EpisodeLayout) -> str:
    """Return ``'rig'`` or ``'projection'``: whichever visual upstream succeeded."""
    problems: list[str] = []
    for stage in VISUAL_UPSTREAM_STAGES:
        try:
            _require_succeeded_manifest(layout, stage)
            return stage
        except (ValueError, FileNotFoundError) as exc:
            problems.append(f"{stage}: {exc}")
    raise ValueError(
        "no reviewed visual upstream: neither 'rig' (two-arm) nor 'projection' "
        "(single-arm) has a succeeded stage manifest -- " + " | ".join(problems)
    )


def _m3_prerequisites(
    ctx: StageContext, cfg: SegmentationConfig
) -> tuple[EpisodeManifest, Path, Path, dict[str, Any]]:
    episode = _episode_manifest(ctx.layout, ctx.episode_id)
    for stage in ("ingest", "hand"):
        _require_succeeded_manifest(ctx.layout, stage)
    upstream = _visual_upstream(ctx.layout)
    if not ctx.layout.hand_npz.is_file():
        raise FileNotFoundError(f"hand sequence is missing: {ctx.layout.hand_npz}")
    video_path = _source_video_path(ctx, episode)
    actual_video_hash = A.file_sha256(video_path)
    if actual_video_hash != episode.source.sha256:
        raise ValueError("episode source video SHA-256 does not match episode.json")
    if upstream == "rig":
        review_path, review = _approved_rig_record(ctx, episode, cfg)
    else:
        review_path, review = _approved_m3_record(ctx, episode, cfg)
    return episode, video_path, review_path, review


def _runtime_problem(runtime: RuntimeConfig) -> str | None:
    required = {
        "python_executable": runtime.python_executable,
        "repo_path": runtime.repo_path,
        "checkpoint_path": runtime.checkpoint_path,
    }
    missing = [name for name, value in required.items() if not isinstance(value, str) or not value.strip()]
    if missing:
        return "configure runtime." + ", runtime.".join(missing) + " before inference"
    if runtime.execution == "wsl" and not runtime.wsl_distribution:
        return "set runtime.wsl_distribution for WSL inference"

    # The configured WSL executable is a Linux path, while repository and
    # checkpoint locations are host paths that the bridge maps into WSL.
    paths_to_check = [runtime.repo_path, runtime.checkpoint_path]
    if runtime.execution == "native":
        paths_to_check.append(runtime.python_executable)
    for value in paths_to_check:
        assert value is not None
        path = Path(value)
        if not path.exists():
            return f"configured runtime path does not exist: {value}"
    if runtime.repo_path and not Path(runtime.repo_path).is_dir():
        return f"runtime.repo_path is not a directory: {runtime.repo_path}"
    if runtime.checkpoint_path and not Path(runtime.checkpoint_path).exists():
        return f"runtime.checkpoint_path does not exist: {runtime.checkpoint_path}"
    return None


def _host_path(value: str | Path) -> Path:
    """Resolve a model result path, translating WSL /mnt/<drive> when returned."""
    path = Path(value)
    if path.exists():
        return path.resolve()
    posix = PurePosixPath(str(value))
    if len(posix.parts) >= 3 and posix.parts[:2] == ("/", "mnt"):
        drive = posix.parts[2]
        if len(drive) == 1 and drive.isalpha():
            translated = Path(f"{drive.upper()}:/").joinpath(*posix.parts[3:])
            if translated.exists():
                return translated.resolve()
    return path.resolve()


def _asset_digest(value: str | None) -> str:
    if not value:
        return "unconfigured"
    path = _host_path(value)
    if path.is_file():
        return A.file_sha256(path)
    if path.is_dir():
        entries = {
            child.relative_to(path).as_posix(): A.file_sha256(child)
            for child in sorted(path.rglob("*"))
            if child.is_file()
        }
        return A.hash_json(entries)
    return "missing"


def _implementation_hashes() -> dict[str, str]:
    module_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[3]
    files = {
        "m5_stages.py": module_dir / "m5_stages.py",
        "m5_preview.py": module_dir / "m5_preview.py",
        "m5_contracts.py": module_dir / "m5_contracts.py",
        "mask_refine.py": module_dir / "mask_refine.py",
        "m5_bridge.py": module_dir / "m5_bridge.py",
        "tools/m5_model_worker.py": repo_root / "tools" / "m5_model_worker.py",
    }
    return {
        name: A.file_sha256(path) if path.is_file() else "missing"
        for name, path in files.items()
    }


def _m5_fingerprint(
    *,
    stage: str,
    inputs: dict[str, Any],
    config: dict[str, Any],
    runtime: RuntimeConfig,
) -> str:
    return A.hash_json(
        {
            "stage": stage,
            "version": "1",
            "inputs": inputs,
            "configuration": config,
            "runtime_checkpoint_sha256": _asset_digest(runtime.checkpoint_path),
            "implementation_sha256": _implementation_hashes(),
        }
    )


def _decode_first_frame(video_path: Path) -> NDArray[np.uint8]:
    from ..ingest.video import read_frame_by_index

    frame = read_frame_by_index(video_path, 0)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("first decoded source frame is not uint8 RGB")
    return frame


def _decode_source_frames(
    video_path: Path,
    episode: EpisodeManifest,
    *,
    expected_video_hash: str,
) -> list[tuple[int, float, NDArray[np.uint8]]]:
    """Decode PyAV PTS and verify every frame against the episode timebase."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to preserve source video PTS; install the video extra") from exc

    if episode.timebase.timebase_source != "pts":
        raise ValueError("M5 requires the source episode PTS timebase")
    expected_ts = np.asarray(episode.timebase.timestamps_s, dtype=np.float64)
    expected_count = int(episode.timebase.num_frames)
    if expected_ts.shape != (expected_count,) or not np.isfinite(expected_ts).all():
        raise ValueError("episode.json timestamps_s do not match timebase.num_frames")
    if A.file_sha256(video_path) != expected_video_hash:
        raise ValueError("source video changed after input validation")

    decoded: list[tuple[int, float, NDArray[np.uint8]]] = []
    previous_timestamp = -math.inf
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError("source video has no video stream")
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise ValueError("source video stream has no PTS time base")
        time_base = float(stream.time_base)
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"source frame {frame_index} has no PTS")
            timestamp_s = float(frame.pts) * time_base
            if not math.isfinite(timestamp_s) or timestamp_s <= previous_timestamp:
                raise ValueError(f"source PTS is not strictly increasing at frame {frame_index}")
            if frame_index >= expected_count:
                raise ValueError("source video decoded more frames than episode.json")
            image = frame.to_ndarray(format="rgb24")
            if image.dtype != np.uint8 or image.shape != (episode.camera.height, episode.camera.width, 3):
                raise ValueError(
                    f"source frame {frame_index} dimensions {image.shape} do not match "
                    f"episode camera {(episode.camera.height, episode.camera.width)}"
                )
            decoded.append((frame_index, timestamp_s, image))
            previous_timestamp = timestamp_s

    if len(decoded) != expected_count:
        raise ValueError(f"source video has {len(decoded)} frames; episode.json declares {expected_count}")
    actual_ts = np.asarray([frame[1] for frame in decoded], dtype=np.float64)
    if not np.allclose(actual_ts, expected_ts, rtol=0.0, atol=1e-9):
        raise ValueError("decoded source PTS timestamps differ from episode.json timebase")
    return decoded


def _validate_hand_timebase(layout: A.EpisodeLayout, decoded: list[tuple[int, float, NDArray[np.uint8]]]) -> None:
    hand = HandSequence.load_npz(layout.hand_npz)
    # ADR-0026: the human-removal path never indexes a hand (SAM 3 is prompted with
    # text, not with keypoints); the hand stream is only used as an independent
    # timebase witness.  So both hands are checked, and H=2 is no longer refused.
    require_sequence_support("segmentation", hand)
    hand.assert_valid(expect_hands=hand.num_hands)
    frame_index = np.asarray(hand["frame_index"], dtype=np.int64)
    timestamp_s = np.asarray(hand["timestamp_s"], dtype=np.float64)
    expected_indices = np.asarray([frame[0] for frame in decoded], dtype=np.int64)
    expected_ts = np.asarray([frame[1] for frame in decoded], dtype=np.float64)
    if not np.array_equal(frame_index, expected_indices):
        raise ValueError("hand frame_index does not exactly match decoded source video")
    if timestamp_s.shape != expected_ts.shape or not np.allclose(
        timestamp_s, expected_ts, rtol=0.0, atol=1e-9
    ):
        raise ValueError("hand timestamp_s does not match decoded source PTS")


def _read_image(path: Path, *, rgb: bool = True) -> NDArray[np.uint8]:
    import imageio.v2 as imageio

    image = np.asarray(imageio.imread(path))
    if image.dtype != np.uint8:
        raise ValueError(f"PNG must decode as uint8: {path}")
    if rgb:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"PNG must be RGB with three channels: {path}")
    else:
        if image.ndim != 2:
            raise ValueError(f"mask PNG must be a single grayscale channel: {path}")
    return image


def _read_binary_mask(path: Path, *, height: int, width: int) -> NDArray[np.bool_]:
    image = _read_image(path, rgb=False)
    if image.shape != (height, width):
        raise ValueError(f"mask dimensions {image.shape} do not match {(height, width)}: {path}")
    values = np.unique(image)
    if values.size and not np.isin(values, (0, 255)).all():
        raise ValueError(f"mask PNG must contain only 0 and 255 values: {path}")
    return image == 255


def _persist_status_diagnostic(
    ctx: StageContext,
    *,
    stage: str,
    first_source_frame: NDArray[np.uint8],
    reason: str,
) -> Path:
    path = ctx.layout.previews_dir / f"m5-{stage}-status.png"
    return write_png(path, status_image(first_source_frame, stage=stage, reason=reason))


def _input_failure_message(ctx: StageContext, *, stage: str, exc: Exception) -> str:
    """Create a status image from the real source when input validation fails."""
    message = f"{type(exc).__name__}: {exc}"
    try:
        episode = _episode_manifest(ctx.layout, ctx.episode_id)
        video_path = _source_video_path(ctx, episode)
        first_source_frame = _decode_first_frame(video_path)
        diagnostic = _persist_status_diagnostic(
            ctx,
            stage=stage,
            first_source_frame=first_source_frame,
            reason=f"Input validation failed: {message}",
        )
        relative = diagnostic.relative_to(ctx.layout.root).as_posix()
        return f"{message}; diagnostic image: {relative}"
    except Exception as diagnostic_exc:
        return (
            f"{message}; status image could not be created from a real source frame: "
            f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
        )


def _raise_with_status(
    ctx: StageContext,
    *,
    stage: str,
    first_source_frame: NDArray[np.uint8],
    reason: str,
    cause: BaseException | None = None,
) -> None:
    path = _persist_status_diagnostic(
        ctx, stage=stage, first_source_frame=first_source_frame, reason=reason
    )
    error = M5StageError(
        f"{reason}; inference did not complete. Review the diagnostic image at "
        f"{path.relative_to(ctx.layout.root).as_posix()} and configure the M5 runtime.",
        diagnostic_outputs=(path,),
    )
    if cause is None:
        raise error
    raise error from cause


def _model_frame_paths(
    model_result: ModelResult,
    *,
    expected_count: int,
    output_dir: Path,
) -> list[Path]:
    if model_result.task not in {"segmentation", "inpainting"}:
        raise ValueError(f"model bridge returned unexpected task {model_result.task!r}")
    if len(model_result.frame_paths) != expected_count:
        raise ValueError(
            f"model bridge returned {len(model_result.frame_paths)} frames; expected {expected_count}"
        )
    output_root = output_dir.resolve()
    paths: list[Path] = []
    seen: set[Path] = set()
    for index, item in enumerate(model_result.frame_paths):
        path = _host_path(item)
        if not path.is_file():
            raise FileNotFoundError(f"model frame {index} is missing: {item}")
        try:
            path.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(f"model frame {index} is outside requested output_dir: {item}") from exc
        if path in seen:
            raise ValueError(f"model bridge returned a duplicate frame path: {item}")
        seen.add(path)
        paths.append(path)
    return paths


def _normalise_model_provenance(
    value: Any,
    *,
    temp_root: Path,
    stage_prefix: str,
    explicit_paths: dict[str, str],
) -> Any:
    """Replace ephemeral temp paths in persisted model provenance with stable paths."""
    if isinstance(value, dict):
        return {
            str(key): _normalise_model_provenance(
                item,
                temp_root=temp_root,
                stage_prefix=stage_prefix,
                explicit_paths=explicit_paths,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalise_model_provenance(
                item,
                temp_root=temp_root,
                stage_prefix=stage_prefix,
                explicit_paths=explicit_paths,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return A.json_safe(value)

    if value in explicit_paths:
        return explicit_paths[value]
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute():
        try:
            candidate = Path(value).resolve()
        except (OSError, RuntimeError, ValueError):
            return value
    elif posix.is_absolute() and len(posix.parts) >= 3 and posix.parts[:2] == ("/", "mnt"):
        drive = posix.parts[2]
        if len(drive) != 1 or not drive.isalpha():
            return value
        try:
            candidate = Path(f"{drive.upper()}:/").joinpath(*posix.parts[3:]).resolve()
        except (OSError, RuntimeError, ValueError):
            return value
    else:
        # Free-form model descriptions, prompts, and notes are not paths.
        return value
    for alias in (str(candidate), candidate.as_posix()):
        if alias in explicit_paths:
            return explicit_paths[alias]
    try:
        relative = candidate.relative_to(temp_root.resolve())
    except (OSError, RuntimeError, ValueError):
        return value
    return (PurePosixPath(stage_prefix) / PurePosixPath(relative.as_posix())).as_posix()


def _write_model_provenance(
    directory: Path,
    *,
    task: str,
    result: ModelResult,
    final_frame_paths: list[str],
    normalised_provenance: dict[str, Any],
) -> Path:
    payload = {
        "schema_version": "ego2robot.m5_model_provenance.v1",
        "task": task,
        "frame_paths": final_frame_paths,
        "provenance": normalised_provenance,
    }
    return A.atomic_write_json(directory / "model-provenance.json", payload)


def _tree_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _populate_outputs(result: StageResult, stage_root: Path) -> None:
    for path in _tree_files(stage_root):
        result.add_output(path)


def _fps(episode: EpisodeManifest) -> float:
    fps = float(episode.timebase.nominal_fps or 0.0)
    return fps if math.isfinite(fps) and fps > 0.0 else 30.0


def _read_segmentation_indices(
    layout: A.EpisodeLayout,
) -> tuple[SourceFrameIndex, MaskSequenceIndex, Path, Path]:
    source_index_path = layout.root / "segmentation" / "source-index.json"
    mask_index_path = layout.root / "segmentation" / "bundle" / "index.json"
    if not source_index_path.is_file() or not mask_index_path.is_file():
        raise FileNotFoundError("successful segmentation source/mask indices are required for inpainting")
    source_index = SourceFrameIndex.model_validate(A.read_json(source_index_path))
    mask_index = MaskSequenceIndex.model_validate(A.read_json(mask_index_path))
    if len(source_index.frames) != len(mask_index.frames):
        raise ValueError("source and mask indices have different frame counts")
    if (source_index.width, source_index.height) != (mask_index.width, mask_index.height):
        raise ValueError("source and mask index dimensions differ")
    if source_index.source_video_sha256 != mask_index.source_video_sha256:
        raise ValueError("source and mask indices refer to different source videos")
    for index, (source_frame, mask_frame) in enumerate(zip(source_index.frames, mask_index.frames, strict=True)):
        if source_frame.frame_index != mask_frame.frame_index or not math.isclose(
            source_frame.timestamp_s, mask_frame.timestamp_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"source and mask indices are misaligned at row {index}")
    return source_index, mask_index, source_index_path, mask_index_path


def _read_indexed_inputs(
    layout: A.EpisodeLayout,
    source_index: SourceFrameIndex,
    mask_index: MaskSequenceIndex,
) -> tuple[list[NDArray[np.uint8]], NDArray[np.bool_]]:
    source_images: list[NDArray[np.uint8]] = []
    refined_masks: list[NDArray[np.bool_]] = []
    for source_frame, mask_frame in zip(source_index.frames, mask_index.frames, strict=True):
        source_path = _episode_relative(layout, source_frame.path)
        if not source_path.is_file() or A.file_sha256(source_path) != source_frame.sha256:
            raise ValueError(f"source frame is missing or checksum-mismatched: {source_frame.path}")
        source = _read_image(source_path)
        if source.shape != (source_index.height, source_index.width, 3):
            raise ValueError(f"source frame has unexpected dimensions: {source_frame.path}")
        raw_path = _episode_relative(layout, mask_frame.raw_path)
        refined_path = _episode_relative(layout, mask_frame.refined_path)
        if not raw_path.is_file() or A.file_sha256(raw_path) != mask_frame.raw_sha256:
            raise ValueError(f"raw mask is missing or checksum-mismatched: {mask_frame.raw_path}")
        if not refined_path.is_file() or A.file_sha256(refined_path) != mask_frame.refined_sha256:
            raise ValueError(f"refined mask is missing or checksum-mismatched: {mask_frame.refined_path}")
        raw_mask = _read_binary_mask(raw_path, height=mask_index.height, width=mask_index.width)
        refined_mask = _read_binary_mask(refined_path, height=mask_index.height, width=mask_index.width)
        if bool(np.any(refined_mask)) != mask_frame.valid:
            raise ValueError(f"mask valid flag does not match pixels: {mask_frame.refined_path}")
        source_images.append(source)
        refined_masks.append(refined_mask)
    return source_images, np.stack(refined_masks, axis=0)


@register_stage
class SegmentationStage:
    """SAM 3 adapter stage plus deterministic raw/refined mask artifacts."""

    name = "segmentation"
    version = "1"

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        result = ValidationResult()
        try:
            _m3_prerequisites(ctx, ctx.config.segmentation)
        except Exception as exc:
            result.add("segmentation.invalid_input", _input_failure_message(ctx, stage=self.name, exc=exc))
        return result

    def fingerprint(self, ctx: StageContext) -> str:
        episode, video_path, review_path, _review = _m3_prerequisites(ctx, ctx.config.segmentation)
        upstream = _visual_upstream(ctx.layout)
        input_hashes = {
            "episode_json": A.file_sha256(ctx.layout.episode_json),
            "source_video": A.file_sha256(video_path),
            "hand_sequence": A.file_sha256(ctx.layout.hand_npz),
            "ingest_manifest": A.file_sha256(ctx.layout.manifest_path("ingest")),
            "hand_manifest": A.file_sha256(ctx.layout.manifest_path("hand")),
            "m3_review": A.file_sha256(review_path),
        }
        # The single-arm key set is kept verbatim so an existing one-hand episode
        # keeps its recorded fingerprint (and does not silently re-run segmentation).
        if upstream == "projection":
            input_hashes["projection_manifest"] = A.file_sha256(ctx.layout.manifest_path("projection"))
        else:
            input_hashes["rig_manifest"] = A.file_sha256(ctx.layout.manifest_path("rig"))
            input_hashes["rig_json"] = A.file_sha256(ctx.layout.rig_json)
        return _m5_fingerprint(
            stage=self.name,
            inputs=input_hashes,
            config=ctx.config.segmentation.model_dump(mode="json"),
            runtime=ctx.config.segmentation.runtime,
        )

    def run(self, ctx: StageContext) -> StageResult:
        episode, video_path, review_path, review = _m3_prerequisites(ctx, ctx.config.segmentation)
        first_source = _decode_first_frame(video_path)
        runtime_problem = _runtime_problem(ctx.config.segmentation.runtime)
        if runtime_problem:
            _raise_with_status(
                ctx,
                stage=self.name,
                first_source_frame=first_source,
                reason=f"SAM 3 runtime is unavailable: {runtime_problem}",
            )

        cfg = ctx.config.segmentation
        expected_video_hash = A.file_sha256(video_path)
        try:
            decoded = _decode_source_frames(video_path, episode, expected_video_hash=expected_video_hash)
            _validate_hand_timebase(ctx.layout, decoded)
            return self._run_model_and_publish(ctx, episode, decoded, expected_video_hash, review_path, review)
        except M5StageError:
            raise
        except Exception as exc:
            _raise_with_status(
                ctx,
                stage=self.name,
                first_source_frame=first_source,
                reason=f"SAM 3 segmentation failed: {type(exc).__name__}: {exc}",
                cause=exc,
            )

    def _run_model_and_publish(
        self,
        ctx: StageContext,
        episode: EpisodeManifest,
        decoded: list[tuple[int, float, NDArray[np.uint8]]],
        source_video_hash: str,
        review_path: Path,
        review: dict[str, Any],
    ) -> StageResult:
        cfg = ctx.config.segmentation
        target = ctx.layout.stage_dir(self.name)
        publisher = A.TempPublish(target)
        stage_result = StageResult(stage=self.name)
        with publisher as temporary_root:
            temporary_root = temporary_root.resolve()
            final_source_records: list[SourceFrame] = []
            worker_source_records: list[SourceFrame] = []
            input_dir = temporary_root / "input"
            for frame_index, timestamp_s, image in decoded:
                relative_path = f"segmentation/input/frame-{frame_index:06d}.png"
                temp_path = temporary_root / "input" / f"frame-{frame_index:06d}.png"
                write_png(temp_path, image)
                digest = A.file_sha256(temp_path)
                final_source_records.append(
                    SourceFrame(
                        frame_index=frame_index,
                        timestamp_s=float(timestamp_s),
                        path=relative_path,
                        sha256=digest,
                    )
                )
                worker_relative = temp_path.relative_to(ctx.layout.root).as_posix()
                worker_source_records.append(
                    SourceFrame(
                        frame_index=frame_index,
                        timestamp_s=float(timestamp_s),
                        path=worker_relative,
                        sha256=digest,
                    )
                )

            worker_source_index = SourceFrameIndex(
                width=episode.camera.width,
                height=episode.camera.height,
                source_video_sha256=source_video_hash,
                frames=worker_source_records,
                provenance={"temporary_model_input": True, "timebase": "source PTS"},
            )
            worker_index_path = temporary_root / "worker-source-index.json"
            A.atomic_write_json(worker_index_path, worker_source_index)

            output_dir = temporary_root / "model-output"
            output_dir.mkdir(parents=True, exist_ok=True)
            request_path = temporary_root / "worker-request.json"
            result_path = temporary_root / "worker-result.json"
            request = ModelRequest(
                task="segmentation",
                episode_root=str(ctx.layout.root.resolve()),
                source_index_path=str(worker_index_path.resolve()),
                output_dir=str(output_dir.resolve()),
                runtime=cfg.runtime,
                parameters={
                    "text_prompt": cfg.text_prompt,
                    "anchor_frame": cfg.anchor_frame,
                    "chunk_frames": cfg.chunk_frames,
                    "chunk_overlap": cfg.chunk_overlap,
                },
            )
            A.atomic_write_json(request_path, request)
            try:
                from .m5_bridge import run_model

                model_result = run_model(request, request_path, result_path)
            except Exception as exc:
                raise RuntimeError(f"SAM 3 model bridge error: {type(exc).__name__}: {exc}") from exc
            if not isinstance(model_result, ModelResult):
                model_result = ModelResult.model_validate(model_result)
            if model_result.task != "segmentation":
                raise ValueError("model bridge returned a non-segmentation result")
            model_paths = _model_frame_paths(
                model_result,
                expected_count=len(decoded),
                output_dir=output_dir,
            )

            raw_masks_list: list[NDArray[np.bool_]] = []
            raw_dir = temporary_root / "bundle" / "masks" / "raw"
            refined_dir = temporary_root / "bundle" / "masks" / "refined"
            for (frame_index, _timestamp_s, _source), model_path in zip(decoded, model_paths, strict=True):
                mask = _read_binary_mask(model_path, height=episode.camera.height, width=episode.camera.width)
                destination = raw_dir / f"frame-{frame_index:06d}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if model_path.resolve() != destination.resolve():
                    shutil.move(str(model_path), str(destination))
                raw_masks_list.append(mask)
            raw_masks = np.stack(raw_masks_list, axis=0)
            refined_masks, refinement_metrics = refine_masks(
                raw_masks,
                gap_fill_max_frames=cfg.gap_fill_max_frames,
                area_median_window=cfg.area_median_window,
                area_median_fraction=cfg.area_median_fraction,
                close_kernel=cfg.close_kernel,
            )

            mask_records: list[MaskFrame] = []
            for index, (frame_index, timestamp_s, _source) in enumerate(decoded):
                raw_path = raw_dir / f"frame-{frame_index:06d}.png"
                refined_path = refined_dir / f"frame-{frame_index:06d}.png"
                write_png(refined_path, refined_masks[index])
                mask_records.append(
                    MaskFrame(
                        frame_index=frame_index,
                        timestamp_s=float(timestamp_s),
                        valid=bool(np.any(refined_masks[index])),
                        raw_path=f"segmentation/bundle/masks/raw/{raw_path.name}",
                        raw_sha256=A.file_sha256(raw_path),
                        refined_path=f"segmentation/bundle/masks/refined/{refined_path.name}",
                        refined_sha256=A.file_sha256(refined_path),
                    )
                )

            final_source_index = SourceFrameIndex(
                width=episode.camera.width,
                height=episode.camera.height,
                source_video_sha256=source_video_hash,
                frames=final_source_records,
                provenance={
                    "source_video_uri": str(episode.source.video_uri),
                    "decoder": "PyAV",
                    "timebase": "PTS",
                    "m3_review": review_path.relative_to(ctx.layout.root).as_posix(),
                    "m3_review_sha256": A.file_sha256(review_path),
                    "m3_review_accepted": review["accepted"],
                },
            )
            A.atomic_write_json(temporary_root / "source-index.json", final_source_index)

            explicit_paths: dict[str, str] = {}
            final_raw_paths: list[str] = []
            for (_, _, _), model_path, record in zip(decoded, model_paths, mask_records, strict=True):
                final_raw_paths.append(record.raw_path)
                explicit_paths[str(model_path)] = record.raw_path
                explicit_paths[model_path.as_posix()] = record.raw_path
            model_provenance = _normalise_model_provenance(
                model_result.provenance,
                temp_root=temporary_root,
                stage_prefix="segmentation",
                explicit_paths=explicit_paths,
            )
            worker_index_sha256 = A.file_sha256(worker_index_path)
            reported_index_sha256 = model_provenance.pop("source_index_sha256", None)
            if reported_index_sha256 not in (None, worker_index_sha256):
                raise ValueError("SAM 3 worker source-index SHA-256 differs from the request index")
            model_provenance.pop("source_index_path", None)
            model_provenance["worker_input_source_index_sha256"] = worker_index_sha256
            model_provenance["worker_input_source_index_persisted"] = False
            provenance = {
                "model_name": cfg.model,
                "model_revision": cfg.runtime.revision,
                "runtime": cfg.runtime.model_dump(mode="json"),
                "configuration": cfg.model_dump(mode="json"),
                "parameters": request.parameters,
                "anchor_frame_index": len(decoded) // 2,
                "source_width": episode.camera.width,
                "source_height": episode.camera.height,
                "source_video_sha256": source_video_hash,
                "m3_review_path": review_path.relative_to(ctx.layout.root).as_posix(),
                "m3_review_sha256": A.file_sha256(review_path),
                "refinement": refinement_metrics,
                "model_provenance": model_provenance,
            }
            # Keep only model-created assets that are still present. Ephemeral
            # request/response JSON is deliberately not published with temp paths.
            (temporary_root / "worker-source-index.json").unlink(missing_ok=True)
            request_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)

            mask_index = MaskSequenceIndex(
                width=episode.camera.width,
                height=episode.camera.height,
                source_video_sha256=source_video_hash,
                frames=mask_records,
                provenance=provenance,
            )
            bundle_dir = temporary_root / "bundle"
            A.atomic_write_json(bundle_dir / "index.json", mask_index)
            A.atomic_write_json(bundle_dir / "refinement-metrics.json", refinement_metrics)
            A.atomic_write_json(
                bundle_dir / "model-provenance.json",
                {
                    "schema_version": "ego2robot.m5_model_provenance.v1",
                    "task": "segmentation",
                    "frame_paths": final_raw_paths,
                    "provenance": model_provenance,
                },
            )

            preview_dir = temporary_root / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            source_images = [frame[2] for frame in decoded]
            preview_frames = [
                make_segmentation_triptych(
                    source_images[i],
                    raw_masks[i],
                    refined_masks[i],
                    frame_index=decoded[i][0],
                    timestamp_s=decoded[i][1],
                )
                for i in range(len(decoded))
            ]
            for index in representative_indices(len(decoded), maximum=8):
                frame_index, _timestamp_s, _ = decoded[index]
                write_png(preview_dir / f"segmentation-frame-{frame_index:06d}.png", preview_frames[index])
            write_png(
                preview_dir / "mask-area-curve.png",
                make_area_curve(
                    refinement_metrics["raw_area_pixels"],
                    refinement_metrics["refined_area_pixels"],
                ),
            )
            write_video(preview_dir / "mask-review.mp4", preview_frames, fps=_fps(episode))
            A.atomic_write_json(
                temporary_root / "stage-report.json",
                {
                    "schema_version": "ego2robot.m5_segmentation_report.v1",
                    "episode_id": episode.episode_id,
                    "num_frames": len(decoded),
                    "width": episode.camera.width,
                    "height": episode.camera.height,
                    "source_video_sha256": source_video_hash,
                    "canonical_timebase": "source frame_index and PTS timestamps in source-index.json",
                    "preview_video_fps": _fps(episode),
                    "valid_mask_frames": sum(record.valid for record in mask_records),
                    "refinement": refinement_metrics,
                    "model_provenance_path": "segmentation/bundle/model-provenance.json",
                    "m3_review_path": review_path.relative_to(ctx.layout.root).as_posix(),
                },
            )

            # Map temp paths in every persisted provenance document to their
            # final episode-relative paths before atomic publish.
            for provenance_path in (
                bundle_dir / "model-provenance.json",
                bundle_dir / "index.json",
            ):
                payload = A.read_json(provenance_path)
                payload = _normalise_model_provenance(
                    payload,
                    temp_root=temporary_root,
                    stage_prefix="segmentation",
                    explicit_paths=explicit_paths,
                )
                A.atomic_write_json(provenance_path, payload)

            publisher.publish()

        _populate_outputs(stage_result, target)
        valid_count = sum(record.valid for record in mask_records)
        stage_result.metrics = {
            "num_frames": len(decoded),
            "valid_mask_frames": int(valid_count),
            "valid_mask_ratio": float(valid_count / len(decoded)) if decoded else 0.0,
            "gap_interpolated_frames": len(refinement_metrics["gap_interpolated_frame_indices"]),
            "area_replaced_frames": len(refinement_metrics["area_replaced_frame_indices"]),
            "retained_empty_frames": len(refinement_metrics["retained_empty_frame_indices"]),
            "width": int(episode.camera.width),
            "height": int(episode.camera.height),
        }
        return stage_result

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        validation = ValidationResult()
        if result.metrics.get("num_frames", 0) <= 0:
            validation.add("segmentation.no_frames", "segmentation produced no frames")
        if not result.outputs:
            validation.add("segmentation.no_outputs", "segmentation produced no artifacts")
        for path in result.outputs:
            if not Path(path).is_file():
                validation.add("segmentation.missing_output", f"declared output is missing: {path}")
        return validation


@register_stage
class InpaintingStage:
    """ProPainter adapter stage with PTS-preserving background PNGs/previews."""

    name = "inpainting"
    version = "1"

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        result = ValidationResult()
        try:
            segmentation_manifest = _require_succeeded_manifest(ctx.layout, "segmentation")
            source_index, mask_index, _source_path, _mask_path = _read_segmentation_indices(ctx.layout)
            declared = set(segmentation_manifest.outputs)
            for required in (
                "segmentation/source-index.json",
                "segmentation/bundle/index.json",
            ):
                if required not in declared:
                    raise ValueError(f"segmentation manifest does not declare required output {required}")
            _read_indexed_inputs(ctx.layout, source_index, mask_index)
        except Exception as exc:
            result.add("inpainting.invalid_input", _input_failure_message(ctx, stage=self.name, exc=exc))
        return result

    def fingerprint(self, ctx: StageContext) -> str:
        segmentation_manifest = _require_succeeded_manifest(ctx.layout, "segmentation")
        source_index, mask_index, source_path, mask_path = _read_segmentation_indices(ctx.layout)
        input_hashes = {
            "segmentation_manifest": A.file_sha256(ctx.layout.manifest_path("segmentation")),
            "source_index": A.file_sha256(source_path),
            "mask_index": A.file_sha256(mask_path),
            "source_video_sha256": source_index.source_video_sha256,
            "segmentation_fingerprint": segmentation_manifest.input_fingerprints.get("self", ""),
            "indexed_source_frames": A.hash_json([frame.sha256 for frame in source_index.frames]),
            "indexed_raw_masks": A.hash_json([frame.raw_sha256 for frame in mask_index.frames]),
            "indexed_refined_masks": A.hash_json([frame.refined_sha256 for frame in mask_index.frames]),
        }
        return _m5_fingerprint(
            stage=self.name,
            inputs=input_hashes,
            config=ctx.config.inpainting.model_dump(mode="json"),
            runtime=ctx.config.inpainting.runtime,
        )

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.config.inpainting
        source_index, mask_index, source_index_path, mask_index_path = _read_segmentation_indices(ctx.layout)
        source_images, refined_masks = _read_indexed_inputs(ctx.layout, source_index, mask_index)
        first_source = source_images[0]
        runtime_problem = _runtime_problem(cfg.runtime)
        if runtime_problem:
            _raise_with_status(
                ctx,
                stage=self.name,
                first_source_frame=first_source,
                reason=f"ProPainter runtime is unavailable: {runtime_problem}",
            )

        try:
            return self._run_model_and_publish(
                ctx,
                source_index,
                mask_index,
                source_index_path,
                mask_index_path,
                source_images,
                refined_masks,
            )
        except M5StageError:
            raise
        except Exception as exc:
            _raise_with_status(
                ctx,
                stage=self.name,
                first_source_frame=first_source,
                reason=f"ProPainter inpainting failed: {type(exc).__name__}: {exc}",
                cause=exc,
            )

    def _run_model_and_publish(
        self,
        ctx: StageContext,
        source_index: SourceFrameIndex,
        mask_index: MaskSequenceIndex,
        source_index_path: Path,
        mask_index_path: Path,
        source_images: list[NDArray[np.uint8]],
        refined_masks: NDArray[np.bool_],
    ) -> StageResult:
        cfg = ctx.config.inpainting
        target = ctx.layout.stage_dir(self.name)
        publisher = A.TempPublish(target)
        result = StageResult(stage=self.name)
        with publisher as temporary_root:
            temporary_root = temporary_root.resolve()
            output_dir = temporary_root / "model-output"
            output_dir.mkdir(parents=True, exist_ok=True)
            request_path = temporary_root / "worker-request.json"
            result_path = temporary_root / "worker-result.json"
            request = ModelRequest(
                task="inpainting",
                episode_root=str(ctx.layout.root.resolve()),
                source_index_path=str(source_index_path.resolve()),
                output_dir=str(output_dir.resolve()),
                mask_index_path=str(mask_index_path.resolve()),
                runtime=cfg.runtime,
                parameters=cfg.model_dump(mode="json", exclude={"runtime", "adr"}),
            )
            A.atomic_write_json(request_path, request)
            try:
                from .m5_bridge import run_model

                model_result = run_model(request, request_path, result_path)
            except Exception as exc:
                raise RuntimeError(f"ProPainter model bridge error: {type(exc).__name__}: {exc}") from exc
            if not isinstance(model_result, ModelResult):
                model_result = ModelResult.model_validate(model_result)
            if model_result.task != "inpainting":
                raise ValueError("model bridge returned a non-inpainting result")
            model_paths = _model_frame_paths(
                model_result,
                expected_count=len(source_index.frames),
                output_dir=output_dir,
            )

            background_dir = temporary_root / "frames" / "background"
            records: list[BackgroundFrame] = []
            explicit_paths: dict[str, str] = {}
            final_background_paths: list[str] = []
            for input_path, final_path in (
                (source_index_path, "segmentation/source-index.json"),
                (mask_index_path, "segmentation/bundle/index.json"),
            ):
                explicit_paths[str(input_path.resolve())] = final_path
                explicit_paths[input_path.resolve().as_posix()] = final_path
            preview_only = bool(
                cfg.inference_width is not None
                and (cfg.inference_width != source_index.width or cfg.inference_height != source_index.height)
            )
            inference_resolution = (
                int(cfg.inference_width or source_index.width),
                int(cfg.inference_height or source_index.height),
            )
            for index, (source_frame, mask_frame, model_path) in enumerate(
                zip(source_index.frames, mask_index.frames, model_paths, strict=True)
            ):
                image = _read_image(model_path)
                if image.shape != (source_index.height, source_index.width, 3):
                    raise ValueError(
                        f"ProPainter output frame {index} has dimensions {image.shape}; expected "
                        f"{(source_index.height, source_index.width, 3)}"
                    )
                if not mask_frame.valid and not np.array_equal(image, source_images[index]):
                    raise ValueError(
                        f"ProPainter changed frame {source_frame.frame_index} despite an empty refined mask"
                    )
                path = background_dir / f"frame-{source_frame.frame_index:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                if model_path.resolve() != path.resolve():
                    shutil.move(str(model_path), str(path))
                final_path = f"inpainting/frames/background/{path.name}"
                final_background_paths.append(final_path)
                explicit_paths[str(model_path)] = final_path
                explicit_paths[model_path.as_posix()] = final_path
                records.append(
                    BackgroundFrame(
                        frame_index=source_frame.frame_index,
                        timestamp_s=source_frame.timestamp_s,
                        valid=mask_frame.valid,
                        path=final_path,
                        sha256=A.file_sha256(path),
                    )
                )

            model_provenance = _normalise_model_provenance(
                model_result.provenance,
                temp_root=temporary_root,
                stage_prefix="inpainting",
                explicit_paths=explicit_paths,
            )
            if "preview_only" in model_provenance and bool(model_provenance["preview_only"]) != preview_only:
                raise ValueError("model provenance preview_only flag disagrees with configured dimensions")
            model_provenance = {
                **model_provenance,
                "preview_only": preview_only,
                "source_resolution": [source_index.width, source_index.height],
                "inference_resolution": list(inference_resolution),
                "source_video_sha256": source_index.source_video_sha256,
                "mask_index_sha256": A.file_sha256(mask_index_path),
                "processing_contract": {
                    "rgb_resize": "area when downscaling",
                    "mask_resize": "nearest-neighbour",
                    "restore_resize": "cubic to original source dimensions",
                    "outside_dilated_repair_mask": "preserve source pixels",
                },
                "configuration": cfg.model_dump(mode="json"),
                "runtime": cfg.runtime.model_dump(mode="json"),
            }
            (temporary_root / "worker-request.json").unlink(missing_ok=True)
            (temporary_root / "worker-result.json").unlink(missing_ok=True)

            background_index = BackgroundSequenceIndex(
                width=source_index.width,
                height=source_index.height,
                source_video_sha256=source_index.source_video_sha256,
                frames=records,
                provenance={
                    **model_provenance,
                    "source_index_path": "segmentation/source-index.json",
                    "mask_index_path": "segmentation/bundle/index.json",
                    "canonical_timebase": "source frame_index and PTS timestamps in source-index.json",
                },
            )
            bundle_dir = temporary_root / "bundle"
            A.atomic_write_json(bundle_dir / "index.json", background_index)
            A.atomic_write_json(
                bundle_dir / "model-provenance.json",
                {
                    "schema_version": "ego2robot.m5_model_provenance.v1",
                    "task": "inpainting",
                    "frame_paths": final_background_paths,
                    "provenance": model_provenance,
                },
            )

            preview_dir = temporary_root / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            background_images = [_read_image(background_dir / f"frame-{frame.frame_index:06d}.png") for frame in records]
            preview_frames = [
                make_inpainting_triptych(
                    source_images[i],
                    refined_masks[i],
                    background_images[i],
                    frame_index=source_index.frames[i].frame_index,
                    timestamp_s=source_index.frames[i].timestamp_s,
                    inference_resolution=inference_resolution,
                )
                for i in range(len(records))
            ]
            for index in representative_indices(len(records), maximum=8):
                frame_index = source_index.frames[index].frame_index
                write_png(preview_dir / f"inpainting-frame-{frame_index:06d}.png", preview_frames[index])
            write_video(preview_dir / "background-review.mp4", preview_frames, fps=_fps(_episode_manifest(ctx.layout, ctx.episode_id)))
            A.atomic_write_json(
                temporary_root / "stage-report.json",
                {
                    "schema_version": "ego2robot.m5_inpainting_report.v1",
                    "episode_id": ctx.episode_id,
                    "num_frames": len(records),
                    "source_resolution": [source_index.width, source_index.height],
                    "inference_resolution": list(inference_resolution),
                    "preview_only": preview_only,
                    "valid_background_frames": sum(frame.valid for frame in records),
                    "canonical_timebase": "source frame_index and PTS timestamps in source-index.json",
                    "preview_video_fps": _fps(_episode_manifest(ctx.layout, ctx.episode_id)),
                    "model_provenance_path": "inpainting/bundle/model-provenance.json",
                    "comparison_frames": [
                        source_index.frames[index].frame_index
                        for index in representative_indices(len(records), maximum=8)
                    ],
                },
            )

            for provenance_path in (bundle_dir / "index.json", bundle_dir / "model-provenance.json"):
                payload = A.read_json(provenance_path)
                payload = _normalise_model_provenance(
                    payload,
                    temp_root=temporary_root,
                    stage_prefix="inpainting",
                    explicit_paths=explicit_paths,
                )
                A.atomic_write_json(provenance_path, payload)

            publisher.publish()

        _populate_outputs(result, target)
        valid_count = sum(frame.valid for frame in records)
        result.metrics = {
            "num_frames": len(records),
            "valid_background_frames": int(valid_count),
            "valid_background_ratio": float(valid_count / len(records)) if records else 0.0,
            "source_width": int(source_index.width),
            "source_height": int(source_index.height),
            "inference_width": int(inference_resolution[0]),
            "inference_height": int(inference_resolution[1]),
            "preview_only": preview_only,
        }
        return result

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        validation = ValidationResult()
        if result.metrics.get("num_frames", 0) <= 0:
            validation.add("inpainting.no_frames", "inpainting produced no frames")
        if not result.outputs:
            validation.add("inpainting.no_outputs", "inpainting produced no artifacts")
        for path in result.outputs:
            if not Path(path).is_file():
                validation.add("inpainting.missing_output", f"declared output is missing: {path}")
        return validation
