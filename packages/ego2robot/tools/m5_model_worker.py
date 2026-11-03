"""Run one real SAM 3 or ProPainter request in an isolated Python process.

This module intentionally imports only project contracts before reading the
request. Model frameworks are imported after device selection and only for the
requested model. It does not download checkpoints or substitute mock results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import runpy
import shutil
import subprocess
import sys
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ego2robot.vision.m5_contracts import (  # noqa: E402
    InpaintingConfig,
    MaskSequenceIndex,
    ModelRequest,
    ModelResult,
    SegmentationConfig,
    SourceFrameIndex,
)


_PROPPAINTER_CHECKPOINTS = (
    "raft-things.pth",
    "recurrent_flow_completion.pth",
    "ProPainter.pth",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_revision(repo: Path, configured: str | None) -> str | None:
    actual: str | None = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        completed = None
    if completed is not None and completed.returncode == 0:
        actual = completed.stdout.strip() or None
    if configured and actual and configured.lower() != actual.lower():
        raise ValueError(
            f"runtime.revision={configured!r} does not match repository HEAD {actual!r} "
            f"at {repo}"
        )
    return configured or actual


def _runtime_repo(request: ModelRequest) -> tuple[Path, str | None]:
    if not request.runtime.repo_path:
        raise ValueError("M5 runtime.repo_path is required")
    repo = Path(request.runtime.repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Configured model repository is missing: {repo}")
    return repo, _repo_revision(repo, request.runtime.revision)


def _runtime_checkpoint(request: ModelRequest) -> Path:
    if not request.runtime.checkpoint_path:
        raise ValueError("M5 runtime.checkpoint_path is required")
    checkpoint = Path(request.runtime.checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Configured {request.task} checkpoint path is missing: {checkpoint}. "
            "The worker never downloads model weights."
        )
    return checkpoint


def _configure_visible_device(device_value: str) -> tuple[str, int | None]:
    """Expose exactly the requested CUDA device before importing torch."""
    # PyTorch's strict deterministic mode requires this before CUDA is
    # initialized so cuBLAS GEMM has a deterministic workspace configuration.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    value = device_value.strip().lower()
    if value == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu", None
    match = re.fullmatch(r"cuda(?::([0-9]+))?", value)
    if not match:
        raise ValueError(
            f"Unsupported M5 device {device_value!r}; use 'cuda', 'cuda:N', or 'cpu'"
        )
    requested_index = int(match.group(1) or "0")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None and inherited.strip():
        visible = [item.strip() for item in inherited.split(",") if item.strip()]
        if requested_index >= len(visible):
            raise ValueError(
                f"Requested {device_value!r}, but CUDA_VISIBLE_DEVICES exposes "
                f"only {len(visible)} device(s)"
            )
        selected_device = visible[requested_index]
    else:
        selected_device = str(requested_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_device
    os.environ["IS_MAIN_PROCESS"] = "1"
    return "cuda:0", requested_index


def _load_torch_and_seed(request: ModelRequest, effective_device: str):
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The configured M5 interpreter is missing NumPy or PyTorch; "
            "install the documented isolated model environment."
        ) from exc

    seed = int(request.runtime.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not enable strict deterministic PyTorch algorithms: {exc}"
        ) from exc

    if effective_device.startswith("cuda"):
        if not torch.cuda.is_available() or not torch.backends.cudnn.is_available():
            raise RuntimeError(
                f"M5 runtime requested {request.runtime.device!r}, but this interpreter "
                "cannot access CUDA with cuDNN"
            )
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "The M5 worker expected one selected CUDA device after visibility "
                f"filtering, found {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(0)
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"
    return np, torch, device_name


def _load_source_index(request: ModelRequest) -> tuple[SourceFrameIndex, Path, list[Path]]:
    index_path = Path(request.source_index_path).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Source frame index is missing: {index_path}")
    try:
        index = SourceFrameIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid source frame index {index_path}: {exc}") from exc

    episode_root = Path(request.episode_root).expanduser().resolve()
    if not episode_root.is_dir():
        raise FileNotFoundError(f"Episode root is missing: {episode_root}")
    paths: list[Path] = []
    for position, frame in enumerate(index.frames):
        relative = PurePosixPath(frame.path.replace("\\", "/"))
        path = (episode_root / Path(*relative.parts)).resolve()
        try:
            path.relative_to(episode_root)
        except ValueError as exc:
            raise ValueError(
                f"Source frame path escapes episode_root at index position {position}: "
                f"{frame.path!r}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(
                f"Source frame {frame.frame_index} is missing: {path}"
            )
        actual_hash = _sha256(path)
        if actual_hash != frame.sha256:
            raise ValueError(
                f"Source frame {frame.frame_index} SHA-256 mismatch: "
                f"index={frame.sha256}, actual={actual_hash}"
            )
        paths.append(path)
    return index, episode_root, paths


def _prepare_owned_dir(path: Path, output_root: Path) -> Path:
    root = output_root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Worker output path escapes output_dir: {resolved}") from exc
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_mask(path: Path, mask: Any) -> None:
    from PIL import Image
    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask for {path}, got shape {array.shape}")
    Image.fromarray(array.astype(np.uint8) * 255, mode="L").save(
        path, format="PNG", optimize=False, compress_level=6
    )


def _read_mask(path: Path, expected_width: int, expected_height: int):
    from PIL import Image
    import numpy as np

    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
        if image.size != (expected_width, expected_height):
            raise ValueError(
                f"Mask dimensions do not match index for {path}: "
                f"got {image.size}, expected {(expected_width, expected_height)}"
            )
    return mask > 0


def _base_provenance(
    request: ModelRequest,
    repo: Path,
    revision: str | None,
    checkpoint: Path,
    checkpoint_hashes: dict[str, str],
    torch: Any,
    device_name: str,
) -> dict[str, Any]:
    return {
        "model": request.task,
        "repository_path": str(repo),
        "repository_revision": revision,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hashes,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "device_requested": request.runtime.device,
        "device_name": device_name,
        "seed": int(request.runtime.seed),
        "determinism": {
            "torch_use_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cross_device_bitwise_guarantee": False,
        },
        "parameters": request.parameters,
    }


def _sam3_output_mask(outputs: Any, width: int, height: int, context: str):
    import numpy as np

    if not isinstance(outputs, dict) or "out_binary_masks" not in outputs:
        raise ValueError(f"SAM 3 returned no out_binary_masks for {context}")
    masks = np.asarray(outputs["out_binary_masks"])
    object_ids = np.asarray(outputs.get("out_obj_ids", []))
    if masks.dtype != np.bool_:
        raise TypeError(
            f"SAM 3 out_binary_masks for {context} must be bool, got {masks.dtype}"
        )
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError(
            f"SAM 3 out_binary_masks for {context} must have shape "
            f"[objects, {height}, {width}], got {masks.shape}"
        )
    if object_ids.shape != (masks.shape[0],):
        raise ValueError(
            f"SAM 3 out_obj_ids for {context} has shape {object_ids.shape}; "
            f"expected {(masks.shape[0],)}"
        )
    if masks.shape[0] == 0:
        return np.zeros((height, width), dtype=bool)
    return np.any(masks, axis=0)


def _segment(request: ModelRequest) -> ModelResult:
    config = SegmentationConfig.model_validate(
        {**request.parameters, "runtime": request.runtime.model_dump(mode="python")}
    )
    if config.model != "sam3":
        raise ValueError(f"Unsupported segmentation model: {config.model!r}")
    if config.anchor_frame != "middle":
        raise ValueError("The SAM 3 M5 adapter requires anchor_frame='middle'")
    effective_device, _ = _configure_visible_device(request.runtime.device)
    if effective_device == "cpu":
        raise RuntimeError("The pinned SAM 3 video predictor requires CUDA; CPU is unsupported")

    repo, revision = _runtime_repo(request)
    builder_path = repo / "sam3" / "model_builder.py"
    if not builder_path.is_file():
        raise FileNotFoundError(
            f"Configured SAM 3 repository is missing its official model builder: {builder_path}"
        )
    checkpoint = _runtime_checkpoint(request)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM 3 checkpoint_path must name a local checkpoint file: {checkpoint}"
        )
    checkpoint_hash = _sha256(checkpoint)
    source_index, episode_root, source_paths = _load_source_index(request)
    output_root = Path(request.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    segmentation_root = output_root / "segmentation"
    raw_dir = _prepare_owned_dir(segmentation_root / "raw_masks", output_root)

    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))
    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import the official SAM 3 builder from {repo}; "
            "install the pinned repository into the configured isolated interpreter."
        ) from exc
    np, torch, device_name = _load_torch_and_seed(request, effective_device)
    predictor = None
    chunk_summaries: list[dict[str, Any]] = []
    frame_cover_count = [0] * len(source_index.frames)
    frame_area_pixels = [0] * len(source_index.frames)

    try:
        # The pinned video predictor accepts checkpoint_path but not load_from_HF.
        # Supplying a verified local checkpoint prevents its lower-level builder
        # from requesting a gated Hugging Face download.
        predictor = build_sam3_video_predictor(
            checkpoint_path=str(checkpoint),
            bpe_path=None,
            gpus_to_use=[0],
            async_loading_frames=False,
            video_loader_type="cv2",
        )
        num_frames = len(source_index.frames)
        start = 0
        chunk_number = 0
        while start < num_frames:
            end = min(start + config.chunk_frames, num_frames)
            local_masks: dict[int, Any] = {}
            with __import__("tempfile").TemporaryDirectory(
                prefix=f"sam3-chunk-{chunk_number:04d}-",
                dir=segmentation_root,
            ) as temp_name:
                frame_folder = Path(temp_name)
                for local_index, source_position in enumerate(range(start, end)):
                    source_path = source_paths[source_position]
                    with __import__("PIL.Image", fromlist=["Image"]).open(source_path) as image:
                        rgb = image.convert("RGB")
                        if rgb.size != (source_index.width, source_index.height):
                            raise ValueError(
                                f"Source frame {source_index.frames[source_position].frame_index} "
                                f"has size {rgb.size}; index records "
                                f"{(source_index.width, source_index.height)}"
                            )
                        rgb.save(
                            frame_folder / f"{local_index:06d}.png",
                            format="PNG",
                            optimize=False,
                            compress_level=3,
                        )

                started = predictor.handle_request(
                    {
                        "type": "start_session",
                        "resource_path": str(frame_folder),
                        "offload_video_to_cpu": True,
                        "offload_state_to_cpu": True,
                    }
                )
                session_id = started.get("session_id")
                if not session_id:
                    raise RuntimeError(f"SAM 3 failed to start chunk session {chunk_number}")
                try:
                    anchor_local = (end - start) // 2
                    prompt_result = predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": anchor_local,
                            "text": config.text_prompt,
                        }
                    )
                    if int(prompt_result.get("frame_index", -1)) != anchor_local:
                        raise ValueError(
                            f"SAM 3 prompt result frame mismatch in chunk {chunk_number}"
                        )
                    _sam3_output_mask(
                        prompt_result.get("outputs"),
                        source_index.width,
                        source_index.height,
                        f"chunk {chunk_number} anchor",
                    )
                    emitted: set[int] = set()
                    for item in predictor.handle_stream_request(
                        {
                            "type": "propagate_in_video",
                            "session_id": session_id,
                            "propagation_direction": "both",
                        }
                    ):
                        local_frame = int(item["frame_index"])
                        if local_frame < 0 or local_frame >= end - start:
                            raise ValueError(
                                f"SAM 3 emitted out-of-range frame {local_frame} "
                                f"for chunk {chunk_number}"
                            )
                        mask = _sam3_output_mask(
                            item.get("outputs"),
                            source_index.width,
                            source_index.height,
                            f"chunk {chunk_number}, local frame {local_frame}",
                        )
                        if local_frame in local_masks:
                            local_masks[local_frame] |= mask
                        else:
                            local_masks[local_frame] = mask.copy()
                        emitted.add(local_frame)

                    expected_local = set(range(end - start))
                    if emitted != expected_local:
                        missing = sorted(expected_local - emitted)
                        raise RuntimeError(
                            f"SAM 3 propagation for chunk {chunk_number} emitted "
                            f"{len(emitted)}/{end - start} frames; missing {missing[:12]}"
                        )
                finally:
                    predictor.handle_request(
                        {"type": "close_session", "session_id": session_id}
                    )

            for local_index, mask in local_masks.items():
                source_position = start + local_index
                frame_index = source_index.frames[source_position].frame_index
                out_path = raw_dir / f"{frame_index:06d}.png"
                if out_path.exists():
                    with __import__("PIL.Image", fromlist=["Image"]).open(out_path) as image:
                        previous = np.asarray(image.convert("L")) > 0
                    if previous.shape != mask.shape:
                        raise ValueError(
                            f"Stored overlap mask for frame {frame_index} has shape "
                            f"{previous.shape}; SAM returned {mask.shape}"
                        )
                    mask |= previous
                _write_mask(out_path, mask)
                frame_cover_count[source_position] += 1
                frame_area_pixels[source_position] = int(np.count_nonzero(mask))

            chunk_summaries.append(
                {
                    "chunk_index": chunk_number,
                    "start_position": start,
                    "end_position_exclusive": end,
                    "anchor_local_frame": (end - start) // 2,
                    "streamed_frame_count": end - start,
                }
            )
            if end == num_frames:
                break
            start = end - config.chunk_overlap
            chunk_number += 1
    finally:
        if predictor is not None:
            try:
                predictor.shutdown()
            except Exception:
                # Preserve an inference failure; the process itself is isolated
                # and will release CUDA resources on exit.
                pass

    if any(count == 0 for count in frame_cover_count):
        missing = [
            source_index.frames[pos].frame_index
            for pos, count in enumerate(frame_cover_count)
            if count == 0
        ]
        raise RuntimeError(f"SAM 3 did not produce masks for source frames {missing[:12]}")

    frame_paths = [
        str(raw_dir / f"{frame.frame_index:06d}.png") for frame in source_index.frames
    ]
    raw_mask_hashes = {
        str(frame.frame_index): _sha256(Path(path))
        for frame, path in zip(source_index.frames, frame_paths)
    }
    provenance = _base_provenance(
        request,
        repo,
        revision,
        checkpoint,
        {"sam3.pt": checkpoint_hash},
        torch,
        device_name,
    )
    provenance.update(
        {
            "model_name": "SAM 3 video predictor",
            "source_index_path": str(Path(request.source_index_path).resolve()),
            "source_index_sha256": _sha256(Path(request.source_index_path)),
            "source_video_sha256": source_index.source_video_sha256,
            "prompt": config.text_prompt,
            "anchor_frame": config.anchor_frame,
            "chunk_frames": config.chunk_frames,
            "chunk_overlap": config.chunk_overlap,
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
            "video_loader_type": "cv2",
            "output_shape": [len(source_index.frames), source_index.height, source_index.width],
            "frame_index_map": [frame.frame_index for frame in source_index.frames],
            "overlap_merge": "boolean_or",
            "chunk_summaries": chunk_summaries,
            "frame_cover_count": frame_cover_count,
            "raw_mask_area_pixels": frame_area_pixels,
            "raw_mask_sha256": raw_mask_hashes,
            "artifact_paths": {
                "raw_masks": frame_paths,
                "raw_mask_directory": str(raw_dir),
            },
        }
    )
    return ModelResult(task="segmentation", frame_paths=frame_paths, provenance=provenance)


def _load_mask_index(
    request: ModelRequest,
    source_index: SourceFrameIndex,
    episode_root: Path,
) -> tuple[MaskSequenceIndex, list[Path]]:
    if not request.mask_index_path:
        raise ValueError("ProPainter requires request.mask_index_path")
    path = Path(request.mask_index_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mask sequence index is missing: {path}")
    try:
        masks = MaskSequenceIndex.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid mask sequence index {path}: {exc}") from exc
    if len(masks.frames) != len(source_index.frames):
        raise ValueError(
            f"Source/mask frame count mismatch: {len(source_index.frames)} source, "
            f"{len(masks.frames)} masks"
        )
    if masks.width != source_index.width or masks.height != source_index.height:
        raise ValueError(
            f"Source/mask dimensions differ: source={(source_index.width, source_index.height)}, "
            f"mask={(masks.width, masks.height)}"
        )
    if masks.source_video_sha256 != source_index.source_video_sha256:
        raise ValueError("Source and mask indexes refer to different source videos")

    mask_paths: list[Path] = []
    for position, (source_frame, mask_frame) in enumerate(zip(source_index.frames, masks.frames)):
        if source_frame.frame_index != mask_frame.frame_index:
            raise ValueError(
                f"Source/mask frame_index mismatch at position {position}: "
                f"{source_frame.frame_index} vs {mask_frame.frame_index}"
            )
        if not math.isclose(
            source_frame.timestamp_s,
            mask_frame.timestamp_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"Source/mask timestamp mismatch at frame {source_frame.frame_index}: "
                f"{source_frame.timestamp_s} vs {mask_frame.timestamp_s}"
            )
        relative = PurePosixPath(mask_frame.refined_path.replace("\\", "/"))
        mask_path = (episode_root / Path(*relative.parts)).resolve()
        try:
            mask_path.relative_to(episode_root)
        except ValueError as exc:
            raise ValueError(
                f"Refined mask path escapes episode_root at frame {mask_frame.frame_index}"
            ) from exc
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Refined mask for frame {mask_frame.frame_index} is missing: {mask_path}"
            )
        actual_hash = _sha256(mask_path)
        if actual_hash != mask_frame.refined_sha256:
            raise ValueError(
                f"Refined mask SHA-256 mismatch at frame {mask_frame.frame_index}: "
                f"index={mask_frame.refined_sha256}, actual={actual_hash}"
            )
        _read_mask(mask_path, masks.width, masks.height)
        mask_paths.append(mask_path)
    return masks, mask_paths


def _verified_propainter_weights(repo: Path, checkpoint: Path) -> dict[str, tuple[Path, str]]:
    if not checkpoint.is_dir():
        raise ValueError(
            "ProPainter runtime.checkpoint_path must be a directory containing "
            + ", ".join(_PROPPAINTER_CHECKPOINTS)
        )
    official_weights = (repo / "weights").resolve()
    result: dict[str, tuple[Path, str]] = {}
    for filename in _PROPPAINTER_CHECKPOINTS:
        configured_path = checkpoint / filename
        official_path = official_weights / filename
        if not configured_path.is_file():
            raise FileNotFoundError(
                f"Configured ProPainter checkpoint is missing: {configured_path}. "
                "The worker never downloads model weights."
            )
        if not official_path.is_file():
            raise FileNotFoundError(
                f"Official ProPainter loader would auto-download missing weights: "
                f"{official_path}; refusing to run."
            )
        configured_hash = _sha256(configured_path)
        official_hash = _sha256(official_path)
        if configured_hash != official_hash:
            raise ValueError(
                f"Configured ProPainter checkpoint {configured_path} differs from "
                f"the file the official script would load at {official_path}; "
                "place the approved checkpoint files in the official weights directory "
                "or use that directory as runtime.checkpoint_path."
            )
        result[filename] = (official_path, official_hash)
    return result


def _run_official_propainter(
    repo: Path,
    config: InpaintingConfig,
    input_frames: Path,
    input_masks: Path,
    output_dir: Path,
    checkpoint_hashes: dict[str, tuple[Path, str]],
) -> Path:
    script = repo / "inference_propainter.py"
    if not script.is_file():
        raise FileNotFoundError(f"Official ProPainter inference script is missing: {script}")

    # Guard the pinned script's loader so a missing file cannot trigger an
    # implicit network download after the preflight checks above.
    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))
    import utils.download_util as download_util

    allowed = {name: record[0].resolve() for name, record in checkpoint_hashes.items()}

    def verified_local_checkpoint(url: str, model_dir=None, progress=True, file_name=None):
        from urllib.parse import urlparse

        filename = file_name or Path(urlparse(url).path).name
        if filename not in allowed:
            raise RuntimeError(
                f"ProPainter requested an unapproved checkpoint {filename!r}; "
                "automatic downloads are disabled."
            )
        candidate_root = Path(model_dir) if model_dir is not None else repo / "weights"
        candidate = (repo / candidate_root).resolve() if not candidate_root.is_absolute() else candidate_root.resolve()
        candidate = candidate / filename
        if candidate.resolve() != allowed[filename] or not candidate.is_file():
            raise FileNotFoundError(
                f"Verified local ProPainter checkpoint is unavailable: {candidate}; "
                "automatic downloads are disabled."
            )
        return str(candidate)

    download_util.load_file_from_url = verified_local_checkpoint
    argv = [
        str(script),
        "--video",
        str(input_frames),
        "--mask",
        str(input_masks),
        "--output",
        str(output_dir),
        "--width",
        str(config.inference_width if config.inference_width is not None else -1),
        "--height",
        str(config.inference_height if config.inference_height is not None else -1),
        "--neighbor_length",
        str(config.neighbor_length),
        "--ref_stride",
        str(config.ref_stride),
        "--subvideo_length",
        str(config.subvideo_length),
        "--mask_dilation",
        str(config.mask_dilation),
        "--raft_iter",
        str(config.raft_iterations),
        "--save_frames",
    ]
    if config.fp16:
        argv.append("--fp16")

    old_cwd = Path.cwd()
    old_argv = sys.argv
    try:
        os.chdir(repo)
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    # The upstream frame-folder output root uses the input folder basename.
    return output_dir / input_frames.name / "frames"


def _inpaint(request: ModelRequest) -> ModelResult:
    config = InpaintingConfig.model_validate(
        {**request.parameters, "runtime": request.runtime.model_dump(mode="python")}
    )
    if config.model != "propainter":
        raise ValueError(f"Unsupported inpainting model: {config.model!r}")
    effective_device, requested_gpu = _configure_visible_device(request.runtime.device)
    if effective_device == "cpu" and config.fp16:
        raise ValueError(
            "ProPainter disables fp16 on CPU; set fp16=false explicitly or select CUDA"
        )

    repo, revision = _runtime_repo(request)
    checkpoint = _runtime_checkpoint(request)
    if not (repo / "inference_propainter.py").is_file():
        raise FileNotFoundError(
            f"Configured ProPainter repository has no inference_propainter.py: {repo}"
        )
    weights = _verified_propainter_weights(repo, checkpoint)
    checkpoint_hashes = {name: record[1] for name, record in weights.items()}
    source_index, episode_root, source_paths = _load_source_index(request)
    mask_index, mask_paths = _load_mask_index(request, source_index, episode_root)

    source_width, source_height = source_index.width, source_index.height
    target_width = config.inference_width or source_width
    target_height = config.inference_height or source_height
    if target_width > source_width or target_height > source_height:
        raise ValueError(
            "ProPainter preview dimensions may not upscale the source; "
            f"source={(source_width, source_height)}, target={(target_width, target_height)}"
        )
    process_width = target_width - target_width % 8
    process_height = target_height - target_height % 8
    if process_width <= 0 or process_height <= 0:
        raise ValueError(
            f"ProPainter target size {(target_width, target_height)} is too small for its 8-pixel grid"
        )

    output_root = Path(request.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inpainting_root = output_root / "inpainting"
    input_root = _prepare_owned_dir(inpainting_root / "processing_input", output_root)
    input_frames = input_root / "frames"
    input_masks = input_root / "masks"
    input_frames.mkdir(parents=True)
    input_masks.mkdir(parents=True)
    processing_masks_dir = _prepare_owned_dir(
        inpainting_root / "processing_masks", output_root
    )
    dilated_masks_dir = _prepare_owned_dir(
        inpainting_root / "dilated_processing_masks", output_root
    )
    processing_frames_dir = _prepare_owned_dir(
        inpainting_root / "processing_frames", output_root
    )
    restored_masks_dir = _prepare_owned_dir(
        inpainting_root / "restored_dilated_masks", output_root
    )
    restored_frames_dir = _prepare_owned_dir(
        inpainting_root / "restored_frames", output_root
    )
    cli_output = _prepare_owned_dir(inpainting_root / "propainter_cli", output_root)

    import cv2
    import numpy as np
    import scipy
    from PIL import Image
    from scipy import ndimage

    for position, (source_frame, source_path, mask_frame, mask_path) in enumerate(
        zip(source_index.frames, source_paths, mask_index.frames, mask_paths)
    ):
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                f"Could not decode source RGB frame {source_frame.frame_index}: {source_path}"
            )
        if image.shape[:2] != (source_height, source_width):
            raise ValueError(
                f"Source frame {source_frame.frame_index} dimensions are "
                f"{(image.shape[1], image.shape[0])}; expected {(source_width, source_height)}"
            )
        if (target_width, target_height) != (source_width, source_height):
            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        local_name = f"{position:06d}.png"
        if not cv2.imwrite(str(input_frames / local_name), image):
            raise OSError(f"Could not stage ProPainter input frame: {input_frames / local_name}")

        mask = _read_mask(mask_path, source_width, source_height)
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        if (target_width, target_height) != (source_width, source_height):
            mask_image = mask_image.resize(
                (target_width, target_height),
                resample=Image.Resampling.NEAREST,
            )
        mask_path_out = processing_masks_dir / f"{source_frame.frame_index:06d}.png"
        mask_image.save(mask_path_out, format="PNG", optimize=False, compress_level=6)
        if not cv2.imwrite(str(input_masks / local_name), np.asarray(mask_image)):
            raise OSError(f"Could not stage ProPainter input mask: {input_masks / local_name}")

        # Match upstream read_mask: nearest resize to the model's 8-aligned
        # processing grid followed by scipy's default connectivity-1 dilation.
        actual_mask = np.asarray(mask_image.convert("L"))
        if actual_mask.shape != (process_height, process_width):
            actual_mask = np.asarray(
                mask_image.resize(
                    (process_width, process_height),
                    resample=Image.Resampling.NEAREST,
                ).convert("L")
            )
        if config.mask_dilation:
            actual_dilated = ndimage.binary_dilation(
                actual_mask > 0,
                iterations=config.mask_dilation,
            )
        else:
            actual_dilated = actual_mask > 0
        dilated_path = dilated_masks_dir / f"{source_frame.frame_index:06d}.png"
        Image.fromarray(actual_dilated.astype(np.uint8) * 255, mode="L").save(
            dilated_path, format="PNG", optimize=False, compress_level=6
        )
        restored_mask = Image.fromarray(actual_dilated.astype(np.uint8) * 255, mode="L").resize(
            (source_width, source_height),
            resample=Image.Resampling.NEAREST,
        )
        restored_mask_path = restored_masks_dir / f"{source_frame.frame_index:06d}.png"
        restored_mask.save(
            restored_mask_path, format="PNG", optimize=False, compress_level=6
        )

    np, torch, device_name = _load_torch_and_seed(request, effective_device)
    if effective_device.startswith("cuda") and requested_gpu is not None and torch.cuda.device_count() != 1:
        raise RuntimeError("ProPainter CUDA device isolation did not select exactly one GPU")
    output_frames_dir = _run_official_propainter(
        repo,
        config,
        input_frames,
        input_masks,
        cli_output,
        weights,
    )

    frame_paths: list[str] = []
    processing_hashes: dict[str, str] = {}
    restored_hashes: dict[str, str] = {}
    for position, source_frame in enumerate(source_index.frames):
        cli_frame = output_frames_dir / f"{position:04d}.png"
        if not cli_frame.is_file():
            raise FileNotFoundError(
                f"Official ProPainter did not write frame {position}: {cli_frame}"
            )
        with Image.open(cli_frame) as result_image:
            result_rgb = np.asarray(result_image.convert("RGB"))
        if result_rgb.shape[:2] != (target_height, target_width):
            raise ValueError(
                f"ProPainter frame {position} has dimensions "
                f"{(result_rgb.shape[1], result_rgb.shape[0])}; expected "
                f"{(target_width, target_height)}"
            )
        processing_path = processing_frames_dir / f"{source_frame.frame_index:06d}.png"
        Image.fromarray(result_rgb, mode="RGB").save(
            processing_path, format="PNG", optimize=False, compress_level=6
        )
        processing_hashes[str(source_frame.frame_index)] = _sha256(processing_path)

        original = cv2.imread(str(source_paths[position]), cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError(
                f"Could not reload source frame {source_frame.frame_index}: {source_paths[position]}"
            )
        if (target_width, target_height) == (source_width, source_height):
            restored_rgb = result_rgb.copy()
        else:
            restored_rgb = cv2.resize(
                result_rgb,
                (source_width, source_height),
                interpolation=cv2.INTER_CUBIC,
            )
        restored_mask_path = restored_masks_dir / f"{source_frame.frame_index:06d}.png"
        with Image.open(restored_mask_path) as mask_image:
            edit_mask = np.asarray(mask_image.convert("L")) > 0
        if edit_mask.shape != (source_height, source_width):
            raise ValueError(
                f"Restored mask for frame {source_frame.frame_index} has shape {edit_mask.shape}"
            )
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        final_rgb = original_rgb.copy()
        final_rgb[edit_mask] = restored_rgb[edit_mask]
        restored_path = restored_frames_dir / f"{source_frame.frame_index:06d}.png"
        Image.fromarray(final_rgb, mode="RGB").save(
            restored_path, format="PNG", optimize=False, compress_level=6
        )
        frame_paths.append(str(restored_path))
        restored_hashes[str(source_frame.frame_index)] = _sha256(restored_path)

    actual_checkpoint = {name: str(path) for name, (path, _) in weights.items()}
    provenance = _base_provenance(
        request,
        repo,
        revision,
        checkpoint,
        checkpoint_hashes,
        torch,
        device_name,
    )
    provenance.update(
        {
            "model_name": "official ProPainter inference_propainter.py",
            "repository_revision": revision,
            "official_cli_args": {
                "width": target_width,
                "height": target_height,
                "process_width": process_width,
                "process_height": process_height,
                "fp16": config.fp16,
                "neighbor_length": config.neighbor_length,
                "ref_stride": config.ref_stride,
                "subvideo_length": config.subvideo_length,
                "mask_dilation": config.mask_dilation,
                "raft_iter": config.raft_iterations,
                "save_frames": True,
                "video": str(input_frames),
                "mask": str(input_masks),
                "output": str(cli_output),
            },
            "official_weight_paths": actual_checkpoint,
            "source_width": source_width,
            "source_height": source_height,
            "inference_width": target_width,
            "inference_height": target_height,
            "model_processing_width": process_width,
            "model_processing_height": process_height,
            "preview_only": target_width != source_width or target_height != source_height,
            "mask_resize": "nearest",
            "rgb_resize_to_inference": (
                "opencv_inter_area"
                if target_width != source_width or target_height != source_height
                else "identity"
            ),
            "restore_resize": "opencv_inter_cubic",
            "restore_write_region": "upscaled_actual_propainter_dilated_mask_only",
            "propainter_mask_dilation_structure": "scipy_default_connectivity_1",
            "mask_valid": [frame.valid for frame in mask_index.frames],
            "source_video_sha256": source_index.source_video_sha256,
            "mask_index_path": str(Path(request.mask_index_path).resolve()),
            "mask_index_sha256": _sha256(Path(request.mask_index_path)),
            "frame_index_map": [frame.frame_index for frame in source_index.frames],
            "frame_timestamp_s": [frame.timestamp_s for frame in source_index.frames],
            "source_frame_sha256": {
                str(frame.frame_index): digest
                for frame in source_index.frames
                for digest in (frame.sha256,)
            },
            "processing_frame_sha256": processing_hashes,
            "restored_frame_sha256": restored_hashes,
            "scipy_version": scipy.__version__,
            "artifact_paths": {
                "processing_input_frames": [
                    str(input_frames / f"{position:06d}.png")
                    for position in range(len(source_index.frames))
                ],
                "processing_input_masks": [
                    str(input_masks / f"{position:06d}.png")
                    for position in range(len(source_index.frames))
                ],
                "processing_masks": [
                    str(processing_masks_dir / f"{frame.frame_index:06d}.png")
                    for frame in source_index.frames
                ],
                "actual_dilated_processing_masks": [
                    str(dilated_masks_dir / f"{frame.frame_index:06d}.png")
                    for frame in source_index.frames
                ],
                "actual_dilated_restored_masks": [
                    str(restored_masks_dir / f"{frame.frame_index:06d}.png")
                    for frame in source_index.frames
                ],
                "propainter_frames": [
                    str(output_frames_dir / f"{position:04d}.png")
                    for position in range(len(source_index.frames))
                ],
                "processing_frames": [
                    str(processing_frames_dir / f"{frame.frame_index:06d}.png")
                    for frame in source_index.frames
                ],
                "restored_frames": frame_paths,
                "propainter_cli_output_directory": str(cli_output),
            },
        }
    )
    return ModelResult(task="inpainting", frame_paths=frame_paths, provenance=provenance)


def _dispatch(request: ModelRequest) -> ModelResult:
    if request.task == "segmentation":
        return _segment(request)
    if request.task == "inpainting":
        return _inpaint(request)
    raise ValueError(f"Unsupported M5 task: {request.task!r}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    try:
        if not args.request.is_file():
            raise FileNotFoundError(f"M5 request JSON is missing: {args.request}")
        request = ModelRequest.model_validate_json(
            args.request.read_text(encoding="utf-8")
        )
        result = _dispatch(request)
        _atomic_write_json(args.result, result.model_dump(mode="json"))
    except Exception as exc:
        print(f"M5 model worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
